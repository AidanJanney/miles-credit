"""
rechunk_rmom6.py
----------------
Rewrite already-preprocessed ``rmom6_{prognostic,forcing}_<year>.zarr`` stores with a time
chunk of 1, in place.

Why this exists
===============
``preprocess_rmom6.py`` used to default to ``--time-chunk 30``. Zarr can only fetch whole
chunks, and the dataset reads exactly one timestep per ``__getitem__``
(``RegionalMOM6Dataset._extract_field`` -> ``ds.sel(time=t)``), so every single-timestep read
was fetching and zstd-decompressing a 30-timestep chunk -- 1.04 GB per 3D variable -- and
discarding 29/30 of it. Measured on ``rmom6_prognostic_2000.zarr``:

    time chunk 30 :  9.0 s to read one 25-level timestep (uo/vo/thetao/so/SSH)
    time chunk  1 :  0.26 s                                        (~34x faster)

That 30x read amplification made training entirely dataloader-bound: a ``forecast_len: 10``,
``train_batch_size: 16`` iteration issues ~176 single-timestep prognostic reads, i.e. ~1580 s
of read work spread over 4 workers -- which is the ~500 s/iter the 8-experiment sweep was
actually seeing, with the GPUs mostly idle.

Rechunking is a one-time cost (~3 min/year, embarrassingly parallel over years) and needs no
recomputation: it only re-lays-out bytes already on disk, so the regrid/level-selection work
``preprocess_rmom6.py`` did is preserved exactly. New stores written by ``preprocess_rmom6.py``
now default to ``--time-chunk 1`` and do not need this script.

Usage
=====
::

    # one year, in place
    python -u scripts/rechunk_rmom6.py --years 2000

    # all 20 years (see scripts/derecho_rechunk_rmom6.sh for the PBS array version)
    python -u scripts/rechunk_rmom6.py --years 2000-2019

Writes to ``<store>.tmp`` and atomically swaps, so an interrupted run leaves the original
store intact. Already-rechunked years are skipped unless ``--overwrite`` is given.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import numpy as np
import xarray as xr
import zarr

STORES = ("rmom6_prognostic_{year}.zarr", "rmom6_forcing_{year}.zarr")


def _time_chunk(store: str) -> int | None:
    """Return the store's current time-chunk size, or None if it can't be determined.

    Reads each array's own ``zarr.json`` rather than the root's consolidated block, so this
    works whether or not the store carries consolidated metadata.
    """
    for name in sorted(os.listdir(store)):
        array_meta = os.path.join(store, name, "zarr.json")
        if not os.path.isfile(array_meta):
            continue
        with open(array_meta) as f:
            meta = json.load(f)
        grid = meta.get("chunk_grid", {}).get("configuration", {}).get("chunk_shape")
        # Data variables are >= 3D (time, [z1_l,] yh, xh); skip 1D coords like time/xh/yh.
        if grid and len(grid) >= 3:
            return grid[0]
    return None


def _verify(store: str, tmp: str, name: str, n_probe: int = 6) -> None:
    """Compare *n_probe* timesteps of every variable in *tmp* against *store*, bit for bit.

    This runs before the original is deleted, so a failure leaves the source untouched. It is
    not paranoia: an earlier version of this script produced a store whose 3D variables were
    silently 100% NaN (numpy-backed ``compute=False`` wrote nothing, and Zarr does not
    materialise all-fill chunks, so nothing errored and nothing was on disk to notice).
    Comparing one timestep at a time keeps peak memory at a couple of hundred MB.
    """
    old = xr.open_dataset(store, engine="zarr", consolidated=False)
    new = xr.open_dataset(tmp, engine="zarr", consolidated=True)
    try:
        if not np.array_equal(old["time"].values, new["time"].values):
            raise ValueError(f"{name}: time axis changed during rechunk.")

        n_time = old.sizes["time"]
        probes = sorted({int(i) for i in np.linspace(0, n_time - 1, min(n_probe, n_time))})
        for var in old.data_vars:
            for i in probes:
                a = old[var].isel(time=i).values
                b = new[var].isel(time=i).values
                if not np.array_equal(a, b, equal_nan=True):
                    raise ValueError(f"{name}: {var} differs at timestep {i}; refusing to publish. {tmp} kept.")
        print(f"[{name}] verified {len(probes)} timesteps x {len(old.data_vars)} vars: bit-identical")
    finally:
        old.close()
        new.close()


def rechunk_store(store: str, time_chunk: int, slab: int | None, overwrite: bool) -> None:
    """Rewrite *store* with a time chunk of *time_chunk*, in place and atomically.

    Streams the rewrite one *slab* of timesteps at a time via Zarr region writes, so peak
    memory is bounded by the slab (one slab of the 25-level prognostic store is
    ``slab * 25 * 457 * 759 * 4 B`` per 3D variable ~= 35 MB * slab) regardless of how many
    years or timesteps the store holds. Do not be tempted to go back to
    ``ds.chunk({"time": 1}).to_zarr(...)``: dask materialises whole source chunks across all
    5 variables at once there, which is enough to get the process killed.

    *slab* defaults to the source's own time chunk so each source chunk is decompressed
    exactly once; a smaller slab lowers peak memory but re-decompresses source chunks.
    """
    name = os.path.basename(store)
    if not os.path.isdir(store):
        print(f"[skip] {name}: does not exist")
        return

    current = _time_chunk(store)
    if current == time_chunk and not overwrite:
        print(f"[skip] {name}: already time-chunk {time_chunk}")
        return

    t0 = time.perf_counter()
    slab = slab or current or 30
    tmp = store + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)

    # Two things matter about this open:
    #   consolidated=False -- forces a real directory scan. Slower than reading the consolidated
    #     block (irrelevant for a one-time rewrite) but immune to a store whose root zarr.json
    #     carries a *null* consolidated block, which xarray opens as an EMPTY dataset, silently.
    #   chunks=... -- makes the variables dask-backed, which is what lets the compute=False call
    #     below write metadata only. On a numpy-backed (non-dask) dataset, compute=False defers
    #     nothing and to_zarr eagerly writes every variable in full -- which both defeats the
    #     memory bound and is how an earlier version of this script silently produced a store
    #     whose 3D variables were 100% NaN.
    src = xr.open_dataset(store, engine="zarr", consolidated=False, chunks={"time": slab})
    try:
        if not src.indexes["time"].is_monotonic_increasing:
            raise ValueError(f"{name}: time axis is not sorted; refusing to rechunk by region.")

        n_time = src.sizes["time"]
        encoding = {
            var: {"chunks": tuple(min(time_chunk, da.sizes[d]) if d == "time" else da.sizes[d] for d in da.dims)}
            for var, da in src.data_vars.items()
        }

        # Lay down structure + all non-time coords, no data (compute=False).
        print(f"[{name}] rewriting {n_time} timesteps, time-chunk {current} -> {time_chunk} (slab {slab})")
        src.to_zarr(tmp, mode="w", encoding=encoding, compute=False, consolidated=False)

        # Region writes carry only the time-varying variables; everything else was already
        # written above, and xarray rejects non-time-dim variables in a time region.
        timeless = [v for v in src.variables if "time" not in src[v].dims]
        for start in range(0, n_time, slab):
            stop = min(start + slab, n_time)
            block = src.isel(time=slice(start, stop)).drop_vars(timeless).load()
            block.to_zarr(tmp, region={"time": slice(start, stop)}, consolidated=False)
            print(f"[{name}]   timesteps {start}-{stop - 1} written")
    finally:
        src.close()

    zarr.consolidate_metadata(tmp)
    _verify(store, tmp, name)
    shutil.rmtree(store)  # os.replace cannot atomically overwrite a non-empty directory
    os.replace(tmp, store)
    print(f"[done] {name}: time-chunk {current} -> {time_chunk} in {time.perf_counter() - t0:.0f}s")


def parse_years(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(y) for y in spec.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", required=True, help="Year, comma-list, or range (e.g. 2000, 2000,2005, 2000-2019).")
    parser.add_argument(
        "--data-dir",
        default=f"/glade/derecho/scratch/{os.environ.get('USER', '')}/rmom6_regional/preprocessed",
        help="Directory holding rmom6_{prognostic,forcing}_<year>.zarr.",
    )
    parser.add_argument("--time-chunk", type=int, default=1, help="Target time-chunk size (default 1).")
    parser.add_argument(
        "--slab",
        type=int,
        default=None,
        help="Timesteps held in memory per region write. Defaults to the source's own time "
        "chunk (each source chunk then decompressed exactly once). Lower it to cut peak "
        "memory; ~35 MB x slab per 3D variable for the 25-level prognostic store.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Rewrite even if already at the target time chunk.")
    args = parser.parse_args()

    for year in parse_years(args.years):
        for pattern in STORES:
            store = os.path.join(args.data_dir, pattern.format(year=year))
            rechunk_store(store, args.time_chunk, args.slab, args.overwrite)


if __name__ == "__main__":
    main()
