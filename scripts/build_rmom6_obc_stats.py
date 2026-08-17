#!/usr/bin/env python
"""Add open-boundary-condition statistics to the existing rMOM6 scaler artifacts.

The OBC rings are declared as their own gen2 sources so they are standardized by the per-source
``normalize`` preblock with **their own** statistics, before ``ocean_obc_halo`` writes them into
the model grid's halo. That only works if the scaler artifacts actually contain keys for those
sources -- otherwise ``pointwise_scaler`` silently skips them (leaving raw values in the input,
the same class of bug as the unnormalized deptho) and ``bridgescaler_transform`` asserts.

Boundary cells are not distributed like the interior: the northern edge is the open Atlantic and
the eastern edge is the inflow through the Lesser Antilles, so reusing interior statistics would
misplace both the mean and the spread. Hence separate stats rather than borrowing.

Reductions match the interior convention:

* pointwise -> per-(level, along-edge cell) mean/std over **time**, shape ``(nz, N)``, so the
  written array lines up cell-for-cell with the ring the dataset yields.
* levelwise -> scalars per (variable, level) collapsed over time and the along-edge dim, matching
  what ``DStandardScalerTensor`` can express (it broadcasts one statistic per channel).

Writes in place, additively: existing keys are untouched, OBC keys are inserted. Safe to re-run.

Usage::

    python scripts/build_rmom6_obc_stats.py                     # all 4 scaler artifacts
    python scripts/build_rmom6_obc_stats.py --years 2000-2001    # quick check
"""

from __future__ import annotations

import argparse
import glob
import os
from getpass import getuser

import numpy as np
import torch
import xarray as xr

PREP = f"/glade/derecho/scratch/{getuser()}/rmom6_regional/preprocessed"
SCALER_DIR = f"/glade/derecho/scratch/{getuser()}/rmom6_regional/scaler"

# edge -> (source name used in the config, along-edge dim)
EDGES = {"north": ("rMOM6_obc_north", "xh"), "east": ("rMOM6_obc_east", "yh")}
VARS_3D = ["uo", "vo", "thetao", "so"]
VARS_2D = ["SSH"]


def ring_stats(edge: str, years: list[int]) -> dict[str, dict[str, np.ndarray]]:
    """Per-cell mean/std over time for one OBC ring, plus per-level scalars."""
    path = os.path.join(PREP, f"obc_{edge}.zarr")
    ds = xr.open_zarr(path)
    t = ds.time.dt.year
    ds = ds.sel(time=(t >= years[0]) & (t <= years[-1]))
    out: dict[str, dict[str, np.ndarray]] = {}
    for var in VARS_3D + VARS_2D:
        if var not in ds:
            continue
        da = ds[var].astype("float64")
        mu = da.mean(dim="time", skipna=True).values
        sd = da.std(dim="time", ddof=1, skipna=True).values
        # A degenerate std would divide by ~0 and blow the channel up; fall back to 1 there,
        # matching _fill_stat's guard in the interior builders.
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        out[var] = {
            "mu": np.atleast_2d(mu),  # (nz, N) for 3D, (1, N) for 2D
            "sigma": np.atleast_2d(sd),
            # Level-wise scalars: collapse the along-edge dim too.
            "mu_level": np.atleast_1d(np.nanmean(np.atleast_2d(mu), axis=-1)),
            "sigma_level": np.atleast_1d(np.nanmean(np.atleast_2d(sd), axis=-1)),
        }
    return out


def key_for(source: str, var: str) -> str:
    """var_key as the dataset builds it: <source>/<field_type>/<dim>/<var>.

    OBCs are declared under dynamic_forcing so assemble_rollout_batch re-reads them from the
    dataset every rollout step rather than carrying the model's own halo prediction forward.
    """
    dim = "3d" if var in VARS_3D else "2d"
    return f"{source}/dynamic_forcing/{dim}/{var}"


def patch_pointwise(path: str, stats: dict, dry_run: bool) -> None:
    blob = torch.load(path, map_location="cpu")
    added = []
    for edge, (source, _) in EDGES.items():
        for var, s in stats[edge].items():
            k = key_for(source, var)
            blob[k] = {
                "mu": torch.tensor(s["mu"], dtype=torch.float32),
                "sigma": torch.tensor(s["sigma"], dtype=torch.float32),
            }
            added.append(f"{k} {tuple(blob[k]['mu'].shape)}")
    if not dry_run:
        torch.save(blob, path)
    print(f"  {os.path.basename(path)}: +{len(added)} keys -> {len(blob)} total")
    for a in added:
        print(f"      {a}")


def patch_bridgescaler(path: str, stats: dict, dry_run: bool) -> None:
    import json

    from bridgescaler import load_scaler_dict, save_scaler_dict
    from bridgescaler.distributed_tensor import DStandardScalerTensor

    scalers = load_scaler_dict(path)
    added = 0
    for edge, (source, _) in EDGES.items():
        for var, s in stats[edge].items():
            k = key_for(source, var)
            # Populated exactly as build_rmom6_scaler.py._make_scaler does. Every field
            # matters: channels_last must be False or set_channel_dim picks -1 and the
            # transform asserts against the along-edge extent (759/457) instead of the
            # level count; mean_x_/var_x_ must be torch tensors because
            # reshape_to_channels_first calls Tensor.view on them; and x_columns_ must be
            # positional ints to match the interior entries, since extract_x_columns
            # matches a batch's columns against them by identity.
            mu = np.atleast_1d(np.asarray(s["mu_level"], dtype=np.float32))
            sd = np.atleast_1d(np.asarray(s["sigma_level"], dtype=np.float32))
            sc = DStandardScalerTensor(channels_last=False)
            sc.mean_x_ = torch.tensor(mu, dtype=torch.float32)
            sc.var_x_ = torch.tensor(sd**2, dtype=torch.float32)
            sc.x_columns_ = list(range(len(mu)))
            sc._fit = True
            sc.n_ = 1
            # Input-only: the halo is prescribed, never predicted, so a target-side entry
            # would be meaningless (the postblock reads ["target"]).
            scalers["input"].setdefault(source, {})[k] = sc
            added += 1
    if not dry_run:
        save_scaler_dict(scalers, path)
    print(f"  {os.path.basename(path)}: +{added} input-side keys")
    del json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", default="2000-2019")
    ap.add_argument("--scaler-dir", default=SCALER_DIR)
    ap.add_argument("--dry-run", action="store_true", help="Compute and report, write nothing.")
    args = ap.parse_args()

    a, _, b = args.years.partition("-")
    years = list(range(int(a), int(b or a) + 1))
    print(f"years: {years[0]}-{years[-1]}\n")

    stats = {}
    for edge in EDGES:
        stats[edge] = ring_stats(edge, years)
        shown = ", ".join(f"{v}{tuple(s['mu'].shape)}" for v, s in stats[edge].items())
        print(f"obc_{edge}: {shown}")
    print()

    for path in sorted(glob.glob(os.path.join(args.scaler_dir, "ocean_pointwise_*.pt"))):
        patch_pointwise(path, stats, args.dry_run)
    for path in sorted(glob.glob(os.path.join(args.scaler_dir, "ocean_bridgescaler_*.json"))):
        patch_bridgescaler(path, stats, args.dry_run)

    if args.dry_run:
        print("\n(dry run — nothing written)")


if __name__ == "__main__":
    main()
