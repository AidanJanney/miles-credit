#!/usr/bin/env python
"""Build pointwise (per-level, per-pixel) xi-normalization stats for the regional MOM6
ocean pipeline's ``pointwise_scaler`` pre/postblocks.

CREDIT-style residual (xi) tendency normalization (Schreck et al. 2024, arXiv:2411.07814
Sec 3.3): T'' = (T - mu) / (xi * sigma). Unlike scripts/build_rmom6_scaler.py (levelwise,
via bridgescaler), this keeps sigma varying per grid point too, read from
explore_statistics_carib12/stats/stats_pointwise_<var>.nc, since bridgescaler's
DStandardScalerTensor can only broadcast a stat uniformly across H/W (see
credit/preblock/pointwise_stats.py's module docstring for why). xi itself has no spatial
component (see explore_statistics_carib12/compute_xi.py) -- it's read from stats_xi.nc and
broadcast across (y, x) before multiplying into sigma. Pass ``--no-xi`` to skip this and use
plain sigma instead -- the "tendency normalization off" arm of the normalization x
rollout-length x tendency experiment grid (see ``EXPERIMENT_DESIGN.md``).

Output is a single torch-saved file: ``{var_key: {"mu": tensor(nz,H,W), "sigma":
tensor(nz,H,W)}}``, consumed directly by ``credit.preblock.pointwise_scaler`` /
``credit.postblock.pointwise_scaler`` (``stats_path`` arg -- point both pre/postblocks at
the same file, the same convention as the bridgescaler json). Land/seafloor points (NaN or
non-positive std in the source stats) are filled mu=0, sigma=1 so the transform is a no-op
there before ``fill_values``/``ocean_wet_mask`` run downstream.

If ``--levels`` is given, it must exactly match the ``--levels`` used for
``preprocess_rmom6.py`` -- the per-level slice is nearest-matched against the shared z1_l
coordinate, same convention as ``build_rmom6_scaler.py``.

``--level-pairs`` is the counterpart for a ``preprocess_rmom6.py --level-pairs`` run (25
thickness-weighted-merged levels). Reads ``stats_pointwise_<var>_levelpairs.nc`` (built by
``build_rmom6_levelpair_pointwise_stats.py`` directly from the level-paired preprocessed store,
not the raw archive -- see that script's docstring for why native-level stats can't be
reindexed for merged levels) and ``stats_xi_levelpairs.nc``, with no nearest-match ``.sel()``
and **no center-average regrid** -- unlike the raw-archive pointwise stats below, uo/vo in the
level-paired stats are already on the tracer-center A-grid (computed directly from
``rmom6_prognostic_<year>.zarr``, which is already regridded), so this is actually the exact
A-grid statistic rather than the approximation the regrid step below produces. Mutually
exclusive with ``--levels``.

⚠️  Grid quirk (native-level ``--levels``/default path only, NOT ``--level-pairs``):
``stats_pointwise_uo.nc``/``stats_pointwise_vo.nc`` were computed on the RAW archive's native
Arakawa-C face grid (``uo`` on ``xq``, one longer than ``xh``; ``vo`` on ``yq``, one longer than
``yh``), while ``preprocess_rmom6.py`` regrids uo/vo to tracer cell centers (Arakawa A, ``xh``/
``yh``) before writing the prognostic store this config's dataset actually reads. This script
applies the identical center-average regrid (``center[i] = 0.5*(face[i]+face[i+1])``, same
formula as ``preprocess_rmom6.py``'s own ``center_average()``, which this script's approach
predates and motivated) to uo/vo's mean and std fields so the resulting stats land on the same
A-grid as the model input/target tensors -- confirmed necessary (a first attempt without this
raised a shape mismatch: 759 vs 760 in ``pointwise_scaler``'s elementwise divide).

Usage::

    # All 50 levels
    python scripts/build_rmom6_pointwise_scaler.py

    # Matches a preprocess_rmom6.py run done with --levels 3.8,5.1,7.9,9.6,92.3,...
    python scripts/build_rmom6_pointwise_scaler.py --levels 3.8,5.1,7.9,9.6,92.3

    # Matches a preprocess_rmom6.py --level-pairs run (needs stats_pointwise_*_levelpairs.nc
    # built by build_rmom6_levelpair_pointwise_stats.py first)
    python scripts/build_rmom6_pointwise_scaler.py --level-pairs

Dependencies: xarray, numpy, torch.
"""

from __future__ import annotations

import argparse
import os
from getpass import getuser

import numpy as np
import torch
import xarray as xr

DEFAULT_STATS_DIR = "/glade/work/ajanney/RegionalEmulation_v2/explore_statistics_carib12/stats"
SOURCE_NAME = "rMOM6"
LEVEL_VARS = ["thetao", "so", "uo", "vo"]  # covered by stats_pointwise_<var>.nc with a z1_l dim
GLOBAL_VARS = ["SSH"]  # covered by stats_pointwise_SSH.nc, no level dim
# Native face axis needing a center-average regrid before use (None = already tracer-grid),
# matching preprocess_rmom6.py's FORCING_SPEC-style "axis" convention: "X" = last array axis
# (uo on xq), "Y" = second-to-last axis (vo on yq).
REGRID_AXIS: dict[str, str | None] = {"thetao": None, "so": None, "uo": "X", "vo": "Y"}


def _center_average(arr: np.ndarray, axis: str) -> np.ndarray:
    """Arakawa-C face -> tracer-center average along the given (X or Y) axis.

    Equivalent to xgcm's ``grid.interp(da, axis, to="center")`` used by preprocess_rmom6.py
    for the same C->A regrid, without requiring xgcm: ``center[i] = 0.5*(face[i]+face[i+1])``.
    """
    if axis == "X":  # last array axis (uo's xq)
        return 0.5 * (arr[..., :-1] + arr[..., 1:])
    if axis == "Y":  # second-to-last array axis (vo's yq)
        return 0.5 * (arr[..., :-1, :] + arr[..., 1:, :])
    raise ValueError(f"_center_average: axis must be 'X' or 'Y', got {axis!r}")


def parse_levels(spec: str | None) -> list[float] | None:
    if spec is None:
        return None
    return [float(v) for v in spec.split(",")]


def _fill_stat(mean_arr: np.ndarray, std_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NaN (land/seafloor) or non-positive std -> mu=0, sigma=1 so the transform is a no-op."""
    valid = np.isfinite(mean_arr) & np.isfinite(std_arr) & (std_arr > 0)
    mu = np.where(valid, mean_arr, 0.0).astype(np.float32)
    sigma = np.where(valid, std_arr, 1.0).astype(np.float32)
    return mu, sigma


def build_stats_dict(
    stats_dir: str, xi_ds: xr.Dataset | None, levels: list[float] | None, level_pairs: bool = False
) -> tuple[dict, list[str]]:
    """``xi_ds=None`` builds the "tendency normalization off" variant (plain std, no xi).

    ``level_pairs`` reads ``stats_pointwise_<var>_levelpairs.nc`` (built by
    ``build_rmom6_levelpair_pointwise_stats.py``) instead, skips the nearest-match ``.sel()``,
    and skips the C-to-A center-average regrid -- see module docstring.
    """
    suffix = "_levelpairs" if level_pairs else ""
    stats: dict[str, dict[str, torch.Tensor]] = {}
    var_keys: list[str] = []

    for varname in LEVEL_VARS:
        ds = xr.open_dataset(os.path.join(stats_dir, f"stats_pointwise_{varname}{suffix}.nc"))
        mean_da, std_da = ds["mean"], ds["std"]
        if levels is not None:
            mean_da = mean_da.sel(z1_l=levels, method="nearest")
            std_da = std_da.sel(z1_l=levels, method="nearest")
        if xi_ds is not None:
            xi_da = xi_ds[f"{varname}_xi"]
            if levels is not None:
                xi_da = xi_da.sel(z1_l=levels, method="nearest")
            eff_std = std_da.values * xi_da.values[:, None, None]  # xi per-level, broadcast over (y, x)
        else:
            eff_std = std_da.values
        mean_vals = mean_da.values
        axis = None if level_pairs else REGRID_AXIS[varname]
        if axis is not None:  # uo/vo: native C-grid face stats -> tracer-center (A-grid)
            mean_vals = _center_average(mean_vals, axis)
            eff_std = _center_average(eff_std, axis)
        mu, sigma = _fill_stat(mean_vals, eff_std)
        key = f"{SOURCE_NAME}/prognostic/3d/{varname}"
        stats[key] = {"mu": torch.from_numpy(mu), "sigma": torch.from_numpy(sigma)}
        var_keys.append(key)
        ds.close()

    for varname in GLOBAL_VARS:
        ds = xr.open_dataset(os.path.join(stats_dir, f"stats_pointwise_{varname}{suffix}.nc"))
        std_vals = ds["std"].values
        eff_std = std_vals * float(xi_ds[f"{varname}_xi"].values) if xi_ds is not None else std_vals
        mu, sigma = _fill_stat(ds["mean"].values, eff_std)
        key = f"{SOURCE_NAME}/prognostic/2d/{varname}"
        # Add a singleton level dim (nz=1) so shape matches (nz, H, W) like the 3D vars.
        stats[key] = {"mu": torch.from_numpy(mu[None]), "sigma": torch.from_numpy(sigma[None])}
        var_keys.append(key)
        ds.close()

    return stats, var_keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stats-dir", default=DEFAULT_STATS_DIR, help="Directory with stats_pointwise_<var>.nc / stats_xi.nc."
    )
    parser.add_argument(
        "--out",
        default=f"/glade/derecho/scratch/{getuser()}/rmom6_regional/scaler/ocean_pointwise_xi.pt",
        help="Output path for the combined pointwise stats file.",
    )
    parser.add_argument(
        "--levels",
        default=None,
        help="Comma-separated target depths in metres, nearest-matched to z1_l. "
        "Must match the --levels used for preprocess_rmom6.py. Omit for all 50 levels. "
        "Mutually exclusive with --level-pairs.",
    )
    parser.add_argument(
        "--level-pairs",
        action="store_true",
        help="Read stats_pointwise_<var>_levelpairs.nc / stats_xi_levelpairs.nc (built by "
        "build_rmom6_levelpair_pointwise_stats.py / build_rmom6_levelpair_stats.py) instead, "
        "matching a preprocess_rmom6.py --level-pairs run's 25 thickness-merged levels. No "
        "nearest-match subsetting, no C-to-A regrid. Mutually exclusive with --levels.",
    )
    parser.add_argument(
        "--no-xi",
        action="store_true",
        help="Skip xi tendency scaling -- sigma stays exactly as stats_pointwise_<var>.nc "
        "reports it, with no per-level tendency-variance rescaling. Default: xi applied.",
    )
    args = parser.parse_args()

    if args.level_pairs and args.levels:
        parser.error("--levels and --level-pairs are mutually exclusive")
    levels = parse_levels(args.levels)
    xi_suffix = "_levelpairs" if args.level_pairs else ""
    xi_ds = None if args.no_xi else xr.open_dataset(os.path.join(args.stats_dir, f"stats_xi{xi_suffix}.nc"))
    stats, var_keys = build_stats_dict(args.stats_dir, xi_ds, levels, level_pairs=args.level_pairs)
    if xi_ds is not None:
        xi_ds.close()

    out = args.out
    if args.level_pairs and out == parser.get_default("out"):
        out = out.replace("ocean_pointwise_xi.pt", "ocean_pointwise_xi_levelpairs.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(stats, out)

    print(f"stats dir : {args.stats_dir}")
    print(f"tendency  : {'off (--no-xi)' if args.no_xi else 'on (xi-scaled)'}")
    print(
        f"levels    : {'pairwise thickness-average (50 -> 25)' if args.level_pairs else ('all 50' if levels is None else levels)}"
    )
    print(f"wrote     : {out}")
    print()
    print("Variables covered (use this exact list for both preblocks.normalize.args.variables")
    print("and postblocks.denorm.args.variables in the pointwise experiment configs):")
    for key in var_keys:
        print(f"  - {key}")
    print()
    print("NOT covered (no stats yet): dynamic_forcing (taux, tauy, net_heat_surface, runoff), static (deptho, wet).")
    print("Leave those out of pointwise_scaler's variables list — they pass through unscaled.")


if __name__ == "__main__":
    main()
