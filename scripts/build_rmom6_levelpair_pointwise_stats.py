#!/usr/bin/env python
"""Recompute per-(level, grid-point) mean/std for the ``--level-pairs`` preprocessing scheme,
sourced from the already-preprocessed ``rmom6_prognostic_<year>.zarr`` stores.

Pointwise sibling of ``build_rmom6_levelpair_stats.py`` (read that script's docstring first --
same reasoning applies here for why native-level ``stats_pointwise_<var>.nc`` can't just be
reindexed for level-paired data: a merged level's std needs the covariance between its two
native layers, which isn't recorded anywhere, so every stat is recomputed directly on the
merged data). This script only adds the ``mean``/``std`` reduction that keeps the spatial
(``yh``, ``xh``) dims instead of collapsing them -- matching
``explore_statistics_carib12/config.py``'s ``STATS_POINTWISE`` reduction (``["time"]`` only).
xi has no spatial component (see ``compute_xi.py``), so the existing per-level
``stats_xi_levelpairs.nc`` (from ``build_rmom6_levelpair_stats.py``) is reused as-is by
``build_rmom6_pointwise_scaler.py --level-pairs`` -- this script does not recompute xi.

One simplification versus ``explore_statistics_carib12/compute_stats.py``'s pointwise config,
beyond the no-wet-mask-needed point already noted in the levelwise script: no C-to-A grid
regrid is needed for uo/vo either. The raw-archive pointwise stats
(``stats_pointwise_uo.nc``/``vo.nc``) were computed on the native Arakawa-C face grid and then
center-averaged onto the A-grid as a post-hoc approximation by
``build_rmom6_pointwise_scaler.py`` (documented there, and in
``OCEAN_MIGRATION_NOTES.md`` Sec 5, as inexact: ``Var(0.5*(A+B))`` needs ``Cov(A,B)``, which that
shortcut has to assume rather than know). Here, uo/vo are read directly from
``rmom6_prognostic_<year>.zarr``, where they're already regridded to tracer centers (``xh``/
``yh``) by ``preprocess_rmom6.py``'s ``center_average()`` -- so the resulting std is the exact
A-grid statistic, not that approximation, as a side effect of sourcing from the preprocessed
store instead of the raw archive.

Output (under ``--out-dir``, default same directory as the raw-archive stats so
``build_rmom6_pointwise_scaler.py --level-pairs`` can find them alongside the originals) --
matches the raw pipeline's one-file-per-variable convention, each holding generic ``mean``/
``std`` data variables (not ``<var>_mean``, unlike the levelwise script's combined file):
    stats_pointwise_thetao_levelpairs.nc   mean/std, shape (z1_l=25, yh=457, xh=759)
    stats_pointwise_so_levelpairs.nc       mean/std, shape (z1_l=25, yh=457, xh=759)
    stats_pointwise_uo_levelpairs.nc       mean/std, shape (z1_l=25, yh=457, xh=759)
    stats_pointwise_vo_levelpairs.nc       mean/std, shape (z1_l=25, yh=457, xh=759)
    stats_pointwise_SSH_levelpairs.nc      mean/std, shape (yh=457, xh=759)

Usage::

    # Full 20-year range (2000-2019)
    python scripts/build_rmom6_levelpair_pointwise_stats.py --scheduler distributed

    # Quick correctness check on a couple of years, single-threaded (no dask cluster needed)
    python scripts/build_rmom6_levelpair_pointwise_stats.py --years 2000-2001 --scheduler synchronous

Dependencies: xarray, dask, zarr, numpy. dask.distributed only needed for --scheduler distributed.
"""

from __future__ import annotations

import argparse
import os
from getpass import getuser

import dask
import xarray as xr

DEFAULT_PREPROCESSED_DIR = f"/glade/derecho/scratch/{getuser()}/rmom6_regional/preprocessed"
DEFAULT_STATS_DIR = "/glade/work/ajanney/RegionalEmulation_v2/explore_statistics_carib12/stats"
LEVEL_VARS = ["thetao", "so", "uo", "vo"]  # 3D, carry z1_l
GLOBAL_VARS = ["SSH"]  # 2D, no z1_l


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


def open_prognostic(preprocessed_dir: str, years: list[int], time_chunk: int) -> xr.Dataset:
    """Open and concatenate the level-paired rmom6_prognostic_<year>.zarr stores along time.

    time_chunk rechunks (coarsens) the dask read graph from the stores' on-disk chunking
    (time=1, one chunk per timestep -- optimal for training's one-timestep-per-__getitem__
    reads, see preprocess_rmom6.py) into fewer, larger chunks, cutting task-graph overhead for
    this full-series reduction; it does not change what gets read, just how many tasks it's
    split into.
    """
    paths = [os.path.join(preprocessed_dir, f"rmom6_prognostic_{y}.zarr") for y in years]
    missing = [p for p in paths if not os.path.isdir(p)]
    if missing:
        raise FileNotFoundError(
            f"missing prognostic zarr store(s), run preprocess_rmom6.py --level-pairs first: {missing}"
        )
    return xr.open_mfdataset(paths, engine="zarr", combine="by_coords", chunks={"time": time_chunk})


def compute_pointwise(ds: xr.Dataset) -> dict[str, xr.Dataset]:
    """One shared dask pass computing per-(level, y, x) mean/std (ddof=1) for every variable,
    reducing only over time -- matching explore_statistics_carib12/config.py's STATS_POINTWISE.

    Building every variable's lazy graph before a single dask.compute() call (rather than one
    compute() per variable) lets dask deduplicate the shared upstream read/decompress tasks
    instead of rescanning the underlying zarr chunks once per variable.
    """
    lazy: dict[str, xr.Dataset] = {}
    for var in LEVEL_VARS + GLOBAL_VARS:
        da = ds[var]
        lazy[var] = xr.Dataset(
            {
                "mean": da.mean(dim=["time"], skipna=True),
                "std": da.std(dim=["time"], skipna=True, ddof=1),
            }
        )
    (lazy,) = dask.compute(lazy)
    return lazy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--preprocessed-dir",
        default=DEFAULT_PREPROCESSED_DIR,
        help="Directory with rmom6_prognostic_<year>.zarr stores.",
    )
    parser.add_argument(
        "--out-dir", default=DEFAULT_STATS_DIR, help="Directory to write stats_pointwise_<var>_levelpairs.nc into."
    )
    parser.add_argument("--years", default="2000-2019", help="Years to include, e.g. '2000-2019', '2003', '2001,2005'.")
    parser.add_argument(
        "--time-chunk",
        type=int,
        default=30,
        help="Dask read chunk size along time for this reduction (not a Zarr rewrite).",
    )
    parser.add_argument(
        "--scheduler",
        choices=["threads", "synchronous", "distributed"],
        default="threads",
        help="Dask scheduler. 'threads' (default) needs only plain dask, no dask.distributed -- "
        "use this unless you know dask.distributed works in your env (as of writing it does not "
        "in miles-credit-casper: installed dask/distributed versions are mismatched). "
        "'synchronous' is single-threaded, useful for debugging. 'distributed' spins up a local "
        "multi-process cluster (--n-workers/--memory-limit), better parallelism for a full run "
        "but only where dask.distributed actually imports.",
    )
    parser.add_argument(
        "--n-workers", type=int, default=20, help="Worker/thread count (threads and distributed schedulers)."
    )
    parser.add_argument(
        "--memory-limit", default="12GB", help="Per-worker memory limit (--scheduler distributed only)."
    )
    args = parser.parse_args()

    years = parse_years(args.years)
    print(f"preprocessed dir : {args.preprocessed_dir}")
    print(f"out dir          : {args.out_dir}")
    print(f"years            : {years[0]}-{years[-1]} ({len(years)} years)")
    print(
        f"scheduler        : {args.scheduler}"
        + (f" ({args.n_workers} workers)" if args.scheduler != "synchronous" else "")
    )

    client = None
    if args.scheduler == "distributed":
        from dask.distributed import Client, LocalCluster

        cluster = LocalCluster(n_workers=args.n_workers, threads_per_worker=1, memory_limit=args.memory_limit)
        client = Client(cluster)
        print(f"dask dashboard   : {client.dashboard_link}")
    else:
        # See build_rmom6_levelpair_stats.py's main() for why ProgressBar + dt=30 + the
        # tr '\r' '\n' viewing tip -- same reasoning applies verbatim here.
        from dask.diagnostics import ProgressBar

        ProgressBar(dt=30).register()
        if args.scheduler == "threads":
            dask.config.set(scheduler="threads", num_workers=args.n_workers)
        else:
            dask.config.set(scheduler="synchronous")

    ds = open_prognostic(args.preprocessed_dir, years, args.time_chunk)

    print("\nComputing pointwise mean/std...", flush=True)
    pointwise = compute_pointwise(ds)

    os.makedirs(args.out_dir, exist_ok=True)
    for var, var_ds in pointwise.items():
        path = os.path.join(args.out_dir, f"stats_pointwise_{var}_levelpairs.nc")
        var_ds.to_netcdf(path)
        print(f"wrote {path}")

    if client is not None:
        client.close()


if __name__ == "__main__":
    main()
