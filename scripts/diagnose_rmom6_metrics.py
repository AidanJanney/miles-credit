"""How much do land cells distort the logged training metrics?

``credit/metrics.py`` applies no mask: ``acc`` is a centred correlation and ``rmse``/``mse``/
``mae`` are plain means, both taken over every cell of the grid. For the regional MOM6 domain
roughly half of every 3D field is land or below-bathymetry geometry, which ``fill_values`` sets
to a constant 0 in *both* ``y`` and ``y_pred``. Those cells are predicted perfectly by
construction, so they enter every metric as free skill.

``loss.mask_missing_targets`` fixes the loss but not the metrics, so ``train_acc``/``valid_mae``
in ``training_log.csv`` are computed over a domain the loss never saw.

This script takes the real 3D wet mask and a real target field, synthesises a prediction with a
known ocean-only skill, and reports the metric with and without the mask. The gap is the
artifact. Nothing here touches training; it reads only.

    python scripts/diagnose_rmom6_metrics.py
    python scripts/diagnose_rmom6_metrics.py --config config/rmom6_regional_levelwise_singlestep_notendency_final.yml
"""

import argparse
import os

import numpy as np
import torch
import xarray as xr
import yaml

DEFAULT_CONFIG = "config/rmom6_regional_levelwise_singlestep_notendency_final.yml"


def build_wet3d(conf: dict) -> torch.Tensor:
    """Reconstruct the (nz, H, W) wet mask exactly as OceanWetMask does."""
    pb = conf["postblocks"]["per_step"]
    args = next(v["args"] for v in pb.values() if v.get("type") == "ocean_wet_mask")

    st = xr.open_dataset(os.path.expandvars(args["static_path"]))
    deptho = torch.as_tensor(np.asarray(st["deptho"].values), dtype=torch.float32)
    wet = torch.as_tensor(np.asarray(st["wet"].values), dtype=torch.float32)

    zsrc = xr.open_zarr(os.path.expandvars(args["level_source_path"]))
    zl = torch.as_tensor(np.asarray(zsrc[args["level_var"]].values), dtype=torch.float32)

    wet3d = (zl.view(-1, 1, 1) <= torch.nan_to_num(deptho, nan=-1.0).unsqueeze(0)) & (wet.unsqueeze(0) == 1)
    return wet3d.float()


def metrics_pair(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> dict:
    """Compute acc/rmse/mse/mae the way credit/metrics.py does, and again over valid cells only.

    Shapes are (C, H, W); the mask is (C, H, W) with 1 = a cell the dataset defines.
    """
    out = {}

    # --- unmasked: exactly credit/metrics.py's reduction over every dim but the channel ---
    dims = (-2, -1)
    pp = pred - pred.mean(dim=dims, keepdim=True)
    yp = y - y.mean(dim=dims, keepdim=True)
    denom = torch.sqrt((pp**2).sum(dim=dims) * (yp**2).sum(dim=dims)) + 1e-7
    out["acc_unmasked"] = ((pp * yp).sum(dim=dims) / denom).mean().item()
    err = pred - y
    out["rmse_unmasked"] = torch.sqrt((err**2).mean(dim=dims)).mean().item()
    out["mse_unmasked"] = (err**2).mean().item()
    out["mae_unmasked"] = err.abs().mean().item()

    # --- masked: the same statistics over valid cells only ---
    n = mask.sum(dim=dims).clamp(min=1)
    pm = (pred * mask).sum(dim=dims, keepdim=True) / n.unsqueeze(-1).unsqueeze(-1)
    ym = (y * mask).sum(dim=dims, keepdim=True) / n.unsqueeze(-1).unsqueeze(-1)
    pp = (pred - pm) * mask
    yp = (y - ym) * mask
    denom = torch.sqrt((pp**2).sum(dim=dims) * (yp**2).sum(dim=dims)) + 1e-7
    out["acc_masked"] = ((pp * yp).sum(dim=dims) / denom).mean().item()
    err = (pred - y) * mask
    out["rmse_masked"] = torch.sqrt((err**2).sum(dim=dims) / n).mean().item()
    out["mse_masked"] = ((err**2).sum() / mask.sum()).item()
    out["mae_masked"] = (err.abs().sum() / mask.sum()).item()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    conf = yaml.safe_load(open(os.path.expandvars(args.config)))
    wet3d = build_wet3d(conf)
    nz, H, W = wet3d.shape
    valid = wet3d.mean().item()
    print(f"grid {nz} x {H} x {W}   valid (ocean) cells: {100 * valid:.2f}%   land: {100 * (1 - valid):.2f}%")
    print()

    torch.manual_seed(args.seed)
    # A normalized target: unit-variance ocean, land held at exactly 0 as fill_values leaves it.
    y = torch.randn(nz, H, W) * wet3d

    print("A prediction with a known ocean-only skill, scored both ways:")
    print(
        f"{'true ocean corr':>16}  {'acc(logged)':>12}{'acc(masked)':>13}   "
        f"{'mae(logged)':>12}{'mae(masked)':>13}   {'rmse(logged)':>13}{'rmse(masked)':>13}"
    )
    print("-" * 100)
    for rho in [0.99, 0.95, 0.9, 0.75, 0.5, 0.0]:
        noise = torch.randn(nz, H, W)
        pred = (rho * y + np.sqrt(max(1.0 - rho**2, 0.0)) * noise) * wet3d
        m = metrics_pair(pred, y, wet3d)
        print(
            f"{rho:>16.2f}  {m['acc_unmasked']:>12.4f}{m['acc_masked']:>13.4f}   "
            f"{m['mae_unmasked']:>12.4f}{m['mae_masked']:>13.4f}   "
            f"{m['rmse_unmasked']:>13.4f}{m['rmse_masked']:>13.4f}"
        )

    print()
    print("Reading the table: the acc columns agree closely -- land sits at the field mean, so it")
    print("adds little to a centred correlation. The error columns do not: land contributes an")
    print("exact zero to every mean, so the logged mae/rmse understate the ocean error by roughly")
    print("the land fraction. Compare mae across configs only if their land fractions match.")


if __name__ == "__main__":
    main()
