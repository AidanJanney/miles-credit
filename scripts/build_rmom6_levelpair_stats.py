#!/usr/bin/env python
"""Recompute mean/std/xi statistics for the ``--level-pairs`` preprocessing scheme, sourced
from the already-preprocessed ``rmom6_prognostic_<year>.zarr`` stores (not the raw archive).

Why this exists: ``explore_statistics_carib12/{compute_stats,compute_xi}.py`` (consumed by
``build_rmom6_scaler.py``/``build_rmom6_pointwise_scaler.py``) compute ``stats_levelwise.nc``/
``stats_xi.nc``/``stats_pointwise_*.nc`` directly from the RAW native-50-``z1_l``-level archive.
Those stats are only valid for a nearest-match ``--levels`` subset of *native* layers — they do
NOT describe ``preprocess_rmom6.py --level-pairs``' thickness-weighted-averaged levels.

A merged level's **mean** is exactly the same thickness-weighted linear combination
``pairwise_thickness_average()`` already applies, so it *could* in principle be derived from the
two native levels' means. Its **std** (and therefore **xi**, which depends on tendency std)
cannot: for ``M = w_lo*X + w_hi*Y``, ``Var(M) = w_lo^2*Var(X) + w_hi^2*Var(Y) +
2*w_lo*w_hi*Cov(X,Y)``, and ``Cov(X,Y)`` between adjacent native levels isn't recorded anywhere
(the same gap already flagged for the uo/vo C-to-A grid regrid in ``OCEAN_MIGRATION_NOTES.md``).
So every stat here is recomputed from scratch, directly on the merged (level-paired) data — the
same data the model actually trains on — rather than re-derived analytically from the old
native-level stats files.

Mirrors ``explore_statistics_carib12/compute_stats.py`` (global + levelwise mean/std, ddof=1)
and ``compute_xi.py``/``xi_normalization.py`` (CREDIT-style residual/tendency normalization,
Schreck et al. 2024 arXiv:2411.07814 Sec 3.3: xi = sigma(delta T') / geometric-mean(all
sigma(delta T') across variables and levels), where T' is the global-mean/std z-scored field
and delta is a first-order time difference) as closely as possible, so the "xi on vs off"
comparison stays apples-to-apples with the existing n25-subset experiment (including
replicating that script's own std ddof=1 (levelwise/global) vs ddof=0 (xi's internal tendency
std) inconsistency verbatim, rather than silently "fixing" it and producing a xi definition
that no longer matches Schreck et al. 2024 as already implemented here).

Two differences from the raw-archive pipeline, both because the preprocessed Zarr stores are
already clean:
    - No wet-mask (``static.nc``) needed: land cells in ``rmom6_prognostic_<year>.zarr`` are
      already NaN (MOM6's own fill value, decoded on read by ``preprocess_rmom6.py``), so a
      plain ``skipna=True`` reduction does the same job the raw pipeline's explicit
      ``da.where(da < 1e18); da.where(wet == 1)`` masking did.
    - No C-to-A grid regrid needed: uo/vo are already tracer-centered (``xh``/``yh``) in the
      preprocessed store, unlike the raw archive's native C-grid (``xq``/``yq``).

Output (under ``--out-dir``, default same directory as the raw-archive stats so
``build_rmom6_scaler.py --level-pairs`` can find them alongside the originals):
    stats_global_levelpairs.nc     <var>_mean/<var>_std, single scalar, all 5 vars
                                    (uo/vo/thetao/so/SSH) -- SSH's entries are its actual
                                    normalization stat; the 3D vars' are only an intermediate
                                    used to z-score for xi.
    stats_levelwise_levelpairs.nc  <var>_mean/<var>_std per merged z1_l layer (25), uo/vo/thetao/so
    stats_xi_levelpairs.nc         <var>_xi per merged z1_l layer (uo/vo/thetao/so) / scalar (SSH)
                                    + xi_gmean (scalar, the geometric-mean denominator)

Usage::

    # Full 20-year range (2000-2019), matching derecho_preprocess_rmom6.sh's --level-pairs run
    python scripts/build_rmom6_levelpair_stats.py --scheduler distributed

    # Quick correctness check on a couple of years, single-threaded (no dask cluster needed)
    python scripts/build_rmom6_levelpair_stats.py --years 2000-2001 --scheduler synchronous

Dependencies: xarray, dask, zarr, numpy. dask.distributed only needed for --scheduler distributed.
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


def compute_global_and_levelwise(ds: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset]:
    """One shared dask pass computing both the fully-reduced 'global' scalar and the
    per-z1_l-level 'levelwise' mean/std for every variable (mean, std with ddof=1 -- matching
    explore_statistics_carib12/config.py's STATS_REGISTRY), sourced from the level-paired data.

    Building both Datasets' lazy graphs before a single dask.compute() call (rather than one
    compute() per stat, as explore_statistics_carib12/compute_stats.py does) lets dask
    deduplicate the shared upstream read/decompress tasks instead of rescanning the underlying
    zarr chunks once per stat.
    """
    global_vars: dict[str, xr.DataArray] = {}
    levelwise_vars: dict[str, xr.DataArray] = {}

    for var in LEVEL_VARS:
        da = ds[var]
        levelwise_vars[f"{var}_mean"] = da.mean(dim=["time", "yh", "xh"], skipna=True)
        levelwise_vars[f"{var}_std"] = da.std(dim=["time", "yh", "xh"], skipna=True, ddof=1)
        global_vars[f"{var}_mean"] = da.mean(dim=["time", "z1_l", "yh", "xh"], skipna=True)
        global_vars[f"{var}_std"] = da.std(dim=["time", "z1_l", "yh", "xh"], skipna=True, ddof=1)

    for var in GLOBAL_VARS:
        da = ds[var]
        global_vars[f"{var}_mean"] = da.mean(dim=["time", "yh", "xh"], skipna=True)
        global_vars[f"{var}_std"] = da.std(dim=["time", "yh", "xh"], skipna=True, ddof=1)

    levelwise_ds = xr.Dataset(levelwise_vars)
    global_ds = xr.Dataset(global_vars)
    (levelwise_ds, global_ds) = dask.compute(levelwise_ds, global_ds)
    return global_ds, levelwise_ds


def compute_xi(ds: xr.Dataset, global_ds: xr.Dataset) -> xr.Dataset:
    """CREDIT-style xi: z-score each variable with its global (all-levels) mean/std, take the
    std of the first-order time difference (per z1_l level for 3D vars, scalar for SSH), then
    divide by the geometric mean of every one of those tendency-std values across all
    variables and levels. See xi_normalization.py's compute_xi() -- this reproduces its
    algorithm exactly, including using ddof=0 (numpy/xarray's std default) for the tendency std
    itself even though the levelwise/global std above uses ddof=1; that mismatch is already
    baked into the existing xi definition this experiment compares against, not introduced here.
    """
    tendency: dict[str, xr.DataArray] = {}
    for var in LEVEL_VARS + GLOBAL_VARS:
        da = ds[var]
        mu = float(global_ds[f"{var}_mean"].values)
        sigma = float(global_ds[f"{var}_std"].values)
        dz = (da - mu) / (sigma + 1e-30)
        spatial_dims = ["yh", "xh"]
        tendency[var] = dz.diff("time").std(dim=["time"] + spatial_dims, skipna=True)

    (tendency,) = dask.compute(tendency)

    all_vals = []
    for var in LEVEL_VARS:
        all_vals.extend(tendency[var].values.tolist())
    for var in GLOBAL_VARS:
        all_vals.append(float(tendency[var].values))
    all_vals = np.array(all_vals, dtype=np.float64)
    if np.any(all_vals <= 0):
        raise ValueError(f"non-positive sigma(delta T') values found: {all_vals[all_vals <= 0]}")
    gmean = float(np.exp(np.mean(np.log(all_vals))))

    data_vars = {}
    for var in LEVEL_VARS:
        data_vars[f"{var}_xi"] = tendency[var] / gmean
    for var in GLOBAL_VARS:
        data_vars[f"{var}_xi"] = xr.DataArray(float(tendency[var].values) / gmean)
    data_vars["xi_gmean"] = xr.DataArray(gmean)
    return xr.Dataset(data_vars)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--preprocessed-dir",
        default=DEFAULT_PREPROCESSED_DIR,
        help="Directory with rmom6_prognostic_<year>.zarr stores.",
    )
    parser.add_argument("--out-dir", default=DEFAULT_STATS_DIR, help="Directory to write stats_*_levelpairs.nc into.")
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
        # ProgressBar prints a live tick/percentage bar to stdout on every dask.compute() call
        # (works with the threads/synchronous local schedulers; the distributed scheduler has
        # its own dashboard instead, hence the branch above). This is the only way to see
        # progress mid-computation with -j oe -k eod PBS output, since the two compute() calls
        # in compute_global_and_levelwise()/compute_xi() otherwise print nothing until they
        # finish (potentially tens of minutes for a full 20-year run).
        from dask.diagnostics import ProgressBar

        # dt=30: default (1s) writes a '\r'-updated line every second, which floods a
        # non-interactive PBS log (a plain file has no terminal to interpret '\r' as
        # cursor-return, so every update is a literal new chunk); view with
        # `tr '\r' '\n' < rmom6_levelpair_stats.o<jobid> | tail` to see the update history.
        ProgressBar(dt=30).register()
        if args.scheduler == "threads":
            dask.config.set(scheduler="threads", num_workers=args.n_workers)
        else:
            dask.config.set(scheduler="synchronous")

    ds = open_prognostic(args.preprocessed_dir, years, args.time_chunk)

    print("\nComputing global + levelwise mean/std...", flush=True)
    global_ds, levelwise_ds = compute_global_and_levelwise(ds)

    print("Computing xi...", flush=True)
    xi_ds = compute_xi(ds, global_ds)

    os.makedirs(args.out_dir, exist_ok=True)
    global_path = os.path.join(args.out_dir, "stats_global_levelpairs.nc")
    levelwise_path = os.path.join(args.out_dir, "stats_levelwise_levelpairs.nc")
    xi_path = os.path.join(args.out_dir, "stats_xi_levelpairs.nc")
    global_ds.to_netcdf(global_path)
    levelwise_ds.to_netcdf(levelwise_path)
    xi_ds.to_netcdf(xi_path)
    print(f"\nwrote {global_path}")
    print(f"wrote {levelwise_path}")
    print(f"wrote {xi_path}")

    if client is not None:
        client.close()


if __name__ == "__main__":
    main()
