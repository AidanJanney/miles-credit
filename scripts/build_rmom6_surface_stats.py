#!/usr/bin/env python
"""Compute normalization stats for the regional MOM6 **surface (2D) input-only** variables --
the ``dynamic_forcing`` and ``static`` channels that ``build_rmom6_pointwise_scaler.py`` and
``build_rmom6_scaler.py`` previously left uncovered ("NOT covered (no stats yet)").

Why this exists: with no stats, those six channels were concatenated into the model input
tensor raw, while the five prognostics arrived standardized to ~N(0, 1). Measured spread of
what actually reached the network:

    prognostics (normalized)   std ~ 1
    deptho                     std ~ 2276   (max 6000)  <- ~2300x every other channel
    net_heat_surface           std ~ 108    (max 963)   <- ~100x
    taux / tauy                std ~ 0.06               <- ~16x too small
    runoff                     std ~ 0.0002             <- effectively invisible

WxFormer uses ``patch_height/width: 1``, so its cross-embed conv sees these magnitudes
directly and the input is dominated by static bathymetry -- which shows up in rollouts as a
bathymetry-shaped imprint on the surface fields.

Two different reductions are needed, because the two field types differ in what varies:

* ``dynamic_forcing`` (taux, tauy, net_heat_surface, runoff) -- read from
  ``rmom6_forcing_<year>.zarr`` and reduced over **time**, keeping (``yh``, ``xh``). This is
  the same per-(grid-point) reduction ``build_rmom6_levelpair_pointwise_stats.py`` applies to
  the prognostics, so forcing channels end up on the same footing as the fields they force.
* ``static`` (deptho, wet) -- **constant in time**, so the pointwise temporal std is
  identically zero and ``_fill_stat``'s ``std > 0`` guard would collapse it to mu=0/sigma=1,
  i.e. no normalization at all (deptho would still enter at 6000). These are instead reduced
  over **space** to a single scalar mean/std, then broadcast back across the grid so the
  written file keeps the same (``yh``, ``xh``) layout every downstream consumer expects.
  NaN (land) is excluded from the reduction but still receives the broadcast constant, so the
  transform is well-defined there too; land is masked downstream by ``fill_values`` anyway.

Output (under ``--out-dir``, alongside the prognostic stats so the scaler builders find them),
one file per variable holding generic ``mean``/``std`` data variables, each shape
(``yh``=457, ``xh``=759) -- matching the one-file-per-variable convention of
``build_rmom6_levelpair_pointwise_stats.py``::

    stats_pointwise_taux_surface.nc
    stats_pointwise_tauy_surface.nc
    stats_pointwise_net_heat_surface_surface.nc
    stats_pointwise_runoff_surface.nc
    stats_pointwise_deptho_surface.nc
    stats_pointwise_wet_surface.nc

These are level-independent, so a single build serves both the native-level and
``--level-pairs`` scaler paths (hence no ``_levelpairs`` suffix).

Usage::

    # Full 20-year range (2000-2019)
    python scripts/build_rmom6_surface_stats.py

    # Quick correctness check on a couple of years, single-threaded
    python scripts/build_rmom6_surface_stats.py --years 2000-2001 --scheduler synchronous

Then regenerate the combined scaler file, which now picks these up automatically::

    python scripts/build_rmom6_pointwise_scaler.py --level-pairs --no-xi \\
        --out /glade/derecho/scratch/$USER/rmom6_regional/scaler/ocean_pointwise_notendency_levelpairs.pt

Dependencies: xarray, dask, zarr, numpy.
"""

from __future__ import annotations

import argparse
import os
from getpass import getuser

import dask
import numpy as np
import xarray as xr

DEFAULT_PREPROCESSED_DIR = f"/glade/derecho/scratch/{getuser()}/rmom6_regional/preprocessed"
DEFAULT_STATS_DIR = "/glade/work/ajanney/RegionalEmulation_v2/explore_statistics_carib12/stats"
DEFAULT_STATIC_PATH = (
    "/glade/derecho/scratch/ajanney/archive/carib12_runoff_tides_rmax600_f200_gioGlofas_gioNNSM"
    "/ocn/hist/carib12_runoff_tides_rmax600_f200_gioGlofas_gioNNSM.mom6.h.static.nc"
)

FORCING_VARS = ["taux", "tauy", "net_heat_surface", "runoff"]  # time-varying -> reduce over time
STATIC_VARS = ["deptho", "wet"]  # time-constant -> reduce over space


def parse_years(spec: str) -> list[int]:
    """Parse '2000-2019', '2000,2005,2010', or '2007' into a sorted list of years."""
    years: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            years.update(range(int(start), int(end) + 1))
        else:
            years.add(int(part))
    return sorted(years)


def open_forcing(preprocessed_dir: str, years: list[int], time_chunk: int) -> xr.Dataset:
    """Open and concatenate the rmom6_forcing_<year>.zarr stores along time."""
    paths = [os.path.join(preprocessed_dir, f"rmom6_forcing_{y}.zarr") for y in years]
    missing = [p for p in paths if not os.path.isdir(p)]
    if missing:
        raise FileNotFoundError(f"missing forcing zarr store(s), run preprocess_rmom6.py first: {missing}")
    return xr.open_mfdataset(paths, engine="zarr", combine="by_coords", chunks={"time": time_chunk})


def compute_forcing_stats(ds: xr.Dataset) -> dict[str, xr.Dataset]:
    """Per-(grid-point) mean/std (ddof=1) over time, one shared dask pass for all forcing vars.

    Building every variable's lazy graph before a single dask.compute() lets dask deduplicate
    the shared upstream read/decompress tasks instead of rescanning the zarr chunks per variable
    -- same reasoning as build_rmom6_levelpair_pointwise_stats.py's compute_pointwise().
    """
    lazy: dict[str, xr.Dataset] = {}
    for var in FORCING_VARS:
        da = ds[var]
        lazy[var] = xr.Dataset(
            {
                "mean": da.mean(dim=["time"], skipna=True),
                "std": da.std(dim=["time"], skipna=True, ddof=1),
                # Scalar counterparts for the levelwise (bridgescaler) path, reduced over space
                # too. NOT derivable from the pointwise fields above: the total variance also
                # needs the spatial variance of the per-point means, which those don't carry.
                "global_mean": da.mean(dim=["time", "yh", "xh"], skipna=True),
                "global_std": da.std(dim=["time", "yh", "xh"], skipna=True, ddof=1),
            }
        )
    (lazy,) = dask.compute(lazy)
    return lazy


def compute_static_stats(static_path: str) -> dict[str, xr.Dataset]:
    """Scalar spatial mean/std per static variable, broadcast back over (yh, xh).

    A temporal reduction is meaningless here (these fields don't vary in time), so the spread
    used for standardization is the spatial one. The scalar is broadcast to a full 2D field --
    including over land, where the source is NaN -- so the written file matches the (yh, xh)
    layout of the forcing stats and needs no special-casing in the scaler builders.
    """
    out: dict[str, xr.Dataset] = {}
    with xr.open_dataset(static_path) as ds:
        for var in STATIC_VARS:
            da = ds[var]
            mean_val = float(da.mean(skipna=True).values)
            std_val = float(da.std(skipna=True, ddof=1).values)
            if not np.isfinite(std_val) or std_val <= 0:
                raise ValueError(f"static var {var!r} has degenerate spatial std ({std_val}); cannot normalize")
            ones = xr.ones_like(da.astype("float64")).fillna(1.0)
            out[var] = xr.Dataset(
                {
                    "mean": ones * mean_val,
                    "std": ones * std_val,
                    # For statics the levelwise scalar IS this spatial stat -- the pointwise
                    # field above is just the same number broadcast over the grid.
                    "global_mean": xr.DataArray(mean_val),
                    "global_std": xr.DataArray(std_val),
                }
            )
            print(f"  {var:18s} spatial mean={mean_val:12.4f}  std={std_val:12.4f}")
    return out


def write_global_stats(stats: dict[str, xr.Dataset], out_dir: str) -> str:
    """Collect every variable's scalar stats into one stats_global_surface.nc.

    Uses the ``<var>_mean``/``<var>_std`` naming of stats_global*.nc (not the generic
    ``mean``/``std`` of the per-variable pointwise files) so build_rmom6_scaler.py reads it
    exactly the way it already reads SSH's entries.
    """
    combined = xr.Dataset(
        {f"{var}_{stat}": ds[f"global_{stat}"] for var, ds in stats.items() for stat in ("mean", "std")}
    )
    path = os.path.join(out_dir, "stats_global_surface.nc")
    combined.to_netcdf(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--preprocessed-dir", default=DEFAULT_PREPROCESSED_DIR, help="Directory with rmom6_forcing_<year>.zarr stores."
    )
    parser.add_argument("--static-path", default=DEFAULT_STATIC_PATH, help="MOM6 static/geometry file.")
    parser.add_argument(
        "--out-dir", default=DEFAULT_STATS_DIR, help="Directory to write stats_pointwise_<var>_surface.nc into."
    )
    parser.add_argument("--years", default="2000-2019", help="Years to include, e.g. '2000-2019', '2003', '2001,2005'.")
    parser.add_argument(
        "--time-chunk", type=int, default=30, help="Dask read chunk size along time for this reduction."
    )
    parser.add_argument(
        "--scheduler",
        choices=["threads", "synchronous", "distributed"],
        default="threads",
        help="Dask scheduler -- see build_rmom6_levelpair_pointwise_stats.py for the tradeoffs.",
    )
    parser.add_argument("--n-workers", type=int, default=20, help="Worker/thread count.")
    parser.add_argument(
        "--memory-limit", default="12GB", help="Per-worker memory limit (--scheduler distributed only)."
    )
    args = parser.parse_args()

    years = parse_years(args.years)
    print(f"preprocessed dir : {args.preprocessed_dir}")
    print(f"static path      : {args.static_path}")
    print(f"out dir          : {args.out_dir}")
    print(f"years            : {years[0]}-{years[-1]} ({len(years)} years)")

    client = None
    if args.scheduler == "distributed":
        from dask.distributed import Client, LocalCluster

        cluster = LocalCluster(n_workers=args.n_workers, threads_per_worker=1, memory_limit=args.memory_limit)
        client = Client(cluster)
        print(f"dask dashboard   : {client.dashboard_link}")
    else:
        from dask.diagnostics import ProgressBar

        ProgressBar(dt=30).register()
        if args.scheduler == "threads":
            dask.config.set(scheduler="threads", num_workers=args.n_workers)
        else:
            dask.config.set(scheduler="synchronous")

    os.makedirs(args.out_dir, exist_ok=True)

    print("\nStatic variables (spatial reduction):", flush=True)
    stats = compute_static_stats(args.static_path)

    print("\nForcing variables (pointwise reduction over time)...", flush=True)
    stats.update(compute_forcing_stats(open_forcing(args.preprocessed_dir, years, args.time_chunk)))

    print()
    for var, var_ds in stats.items():
        path = os.path.join(args.out_dir, f"stats_pointwise_{var}_surface.nc")
        # Drop the scalar entries here so these files keep the plain mean/std layout the
        # pointwise builder expects; the scalars go to stats_global_surface.nc instead.
        var_ds[["mean", "std"]].to_netcdf(path)
        print(f"wrote {path}")
    print(f"wrote {write_global_stats(stats, args.out_dir)}  (scalars, for the levelwise path)")

    if client is not None:
        client.close()


if __name__ == "__main__":
    main()
