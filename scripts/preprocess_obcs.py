#!/usr/bin/env python
"""Preprocess CrocoDash open-boundary-condition (OBC) segment files into CREDIT-ready Zarr.

Mirrors the original workflow in
``Regional_Ocean_Emulation/archive/Thesis_Archive/DataLoader/CESM_Data_Preprocessing/process_obcs.ipynb``
(per-segment extraction, canonical renaming, chunked Zarr output), generalized to run over
all 4 boundary segments and to support a fast date-limited sample run, mirroring
``preprocess_rmom6.py``'s ``--start-date``/``--end-date``.

Source: ``<obc-dir>/forcing_obc_segment_00{1,2,3,4}.nc`` — for this domain (confirmed by
grid-size matching, see below), segment 001/002/003/004 = south/north/west/east.

Grid quirk (why this needs its own script): MOM6 OBC segment files interleave face (q) and
center (h) points along a SINGLE flattened boundary dimension, instead of the separate
xh/xq (or yh/yq) dims the main archive uses. For a boundary of N center points, MOM6 writes
2N+1 values along the flattened dim, alternating q,h,q,h,...,q (N+1 face values bracketing N
center values). Confirmed against this domain's real grid: north/south segments carry
``nx_segment_00{1,2}`` of length 1519 == 2*759+1 (759 = this archive's ``xh`` size); east/west
segments carry ``ny_segment_00{3,4}`` of length 915 == 2*457+1 (457 = ``yh`` size). Taking
every other point starting at index 1 (``.isel({dim: slice(1, None, 2)})``) recovers exactly
the N center-point values — this is the "stride to select only center points" step, and is
the same slice ``process_obcs.ipynb`` used by hand for its two segments, generalized here to
all four via each segment's own detected long dimension.

Variables kept (canonical rename, matching the main prognostic names) — ``dz_*`` layer
thickness fields are dropped, not needed as CREDIT input:

    u_segment_XXX    -> uo      v_segment_XXX    -> vo
    eta_segment_XXX  -> SSH     salt_segment_XXX -> so     temp_segment_XXX -> thetao

``depth`` (the segment files' level coordinate) was verified identical to the main archive's
``z1_l`` (``np.allclose``), so no interpolation is needed there. Native ``z1_l`` layer depths
(metres, 50 layers) for reference when choosing a ``--levels`` subset:

    0.49, 1.54, 2.65, 3.82, 5.08, 6.44, 7.93, 9.57, 11.40, 13.47, 15.81, 18.50,
    21.60, 25.21, 29.44, 34.43, 40.34, 47.37, 55.76, 65.81, 77.85, 92.33, 109.73,
    130.67, 155.85, 186.13, 222.48, 266.04, 318.13, 380.21, 453.94, 541.09, 643.57,
    763.33, 902.34, 1062.44, 1245.29, 1452.25, 1684.28, 1941.89, 2225.08, 2533.34,
    2865.70, 3220.82, 3597.03, 3992.48, 4405.22, 4833.29, 5274.78, 5727.92

``--level-pairs`` is an alternative to ``--levels``, matching ``preprocess_rmom6.py``: instead
of discarding half the native layers, it thickness-weighted-averages each adjacent pair
(0,1), (2,3), ... down to 25 levels, so every native layer still contributes (see
``pairwise_thickness_average()``). Only applied to the 3D (``z1_l``-carrying) variables —
``SSH`` has no level dimension and passes through unchanged. Mutually exclusive with
``--levels``. If the main prognostic archive was preprocessed with ``--level-pairs``, pass it
here too so the OBC boundary values line up with the same 25 merged levels
``ocean_obc_nudge`` reads against.

⚠️  Time calendar caveat — please double check this against how these files were actually
generated. The segment files declare ``calendar: julian, units: days since 2000-01-01`` (a
known default of the CrocoDash/regional-mom6 OBC-writer tooling, not necessarily a real
astronomical-Julian offset). Taking that calendar label literally would shift every OBC
timestamp ~13 days relative to the main archive's own (``gregorian``) daily timestamps. This
script instead treats the stored values as a plain daily counter from 2000-01-01 — i.e. it
assumes OBC day *k* is the same simulation day as the main archive's day *k* — and, since the
main archive's daily means are timestamped at 12:00 (not midnight), shifts by +12h to match
(toggle with ``--no-noon-offset``). If you know the true intended alignment differs, adjust
``_decode_obc_time`` accordingly before trusting a real run.

Usage::

    # 15-day sample, all 4 boundaries -> obc_<direction>_sample.zarr
    python scripts/preprocess_obcs.py --start-date 2000-01-01 --end-date 2000-01-15

    # Full available range (2000-01-01 .. ~2020-01-01) -> obc_<direction>.zarr
    python scripts/preprocess_obcs.py

    # Pairwise thickness-weighted levels (50 -> 25), matching a --level-pairs prognostic run
    python scripts/preprocess_obcs.py --level-pairs

Dependencies: xarray, dask, zarr, numpy, pandas.
"""

from __future__ import annotations

import argparse
import os
import shutil
from getpass import getuser

import numpy as np
import pandas as pd
import xarray as xr
import zarr

DEFAULT_OBC_DIR = "/glade/work/ajanney/CrocoDash_Input/CrocCaribGio/ocnice"
# segment id -> (direction, long-dim prefix). Confirmed by exact grid-size match against
# this domain's xh (759, nx segments) / yh (457, ny segments) — see module docstring.
SEGMENTS: dict[str, tuple[str, str]] = {
    "001": ("south", "nx"),
    "002": ("north", "nx"),
    "003": ("west", "ny"),
    "004": ("east", "ny"),
}
VAR_RENAME = {"u": "uo", "v": "vo", "eta": "SSH", "salt": "so", "temp": "thetao"}


def parse_levels(spec: str | None) -> list[float] | None:
    if spec is None:
        return None
    return [float(v) for v in spec.split(",")]


def _layer_interfaces(z1_l: np.ndarray) -> np.ndarray:
    """Reconstruct MOM6 layer interface depths from cell-center depths (z1_l).

    See ``preprocess_rmom6.py``'s identical helper — duplicated here rather than shared so this
    script stays a standalone, dependency-free CLI. A center is by definition the midpoint of
    its two bracketing interfaces, so interfaces can be recovered top-down: ``z1_i[0] = 0``
    (surface), ``z1_i[k+1] = 2 * z1_l[k] - z1_i[k]``.
    """
    z1_i = np.zeros(len(z1_l) + 1)
    for k, zc in enumerate(z1_l):
        z1_i[k + 1] = 2.0 * zc - z1_i[k]
    return z1_i


def pairwise_thickness_average(ds: xr.Dataset) -> xr.Dataset:
    """Merge adjacent pairs of z1_l levels into one, thickness-weighted: 50 levels -> 25.

    See ``preprocess_rmom6.py``'s identical helper for the full derivation. Every adjacent pair
    (0,1), (2,3), ... is combined into one output layer, weighted by each native layer's
    thickness (from ``_layer_interfaces()``); the merged layer's ``z1_l`` coordinate is the same
    thickness-weighted average applied to depth itself. Callers must pass a Dataset containing
    only ``z1_l``-carrying variables — see ``process_segment``, which splits 3D vars (uo/vo/
    so/thetao) from 2D ``SSH`` before calling this, since multiplying a variable that lacks the
    z1_l dim by the per-level thickness weights would incorrectly broadcast it onto z1_l.
    """
    n = ds.sizes["z1_l"]
    if n % 2 != 0:
        raise ValueError(f"pairwise level averaging requires an even number of z1_l levels, got {n}")

    z1_l = ds["z1_l"].values
    dz = np.diff(_layer_interfaces(z1_l))
    dz_lo, dz_hi = dz[0::2], dz[1::2]
    dz_sum = dz_lo + dz_hi

    # Drop the z1_l coordinate before combining so xarray aligns lower/upper positionally by
    # dimension size, not by label -- the two halves carry disjoint native depths and would
    # otherwise broadcast to all-NaN.
    lower = ds.isel(z1_l=slice(0, None, 2)).drop_vars("z1_l")
    upper = ds.isel(z1_l=slice(1, None, 2)).drop_vars("z1_l")
    w_lo = xr.DataArray(dz_lo, dims="z1_l")
    w_hi = xr.DataArray(dz_hi, dims="z1_l")

    merged = (lower * w_lo + upper * w_hi) / xr.DataArray(dz_sum, dims="z1_l")
    new_z1_l = (z1_l[0::2] * dz_lo + z1_l[1::2] * dz_hi) / dz_sum
    return merged.assign_coords(z1_l=new_z1_l)


def _decode_obc_time(ds: xr.Dataset, noon_offset: bool) -> xr.Dataset:
    """Replace the raw day-count 'time' coordinate with real timestamps.

    See the ⚠️ calendar caveat in the module docstring — this ignores the file's declared
    'julian' calendar and treats the values as a plain daily counter from 2000-01-01.
    """
    day_counts = ds["time"].values.astype(np.float64)
    origin = pd.Timestamp("2000-01-01") + (pd.Timedelta(hours=12) if noon_offset else pd.Timedelta(0))
    times = origin + pd.to_timedelta(day_counts, unit="D")
    return ds.assign_coords(time=times)


def _write_zarr(ds_out: xr.Dataset, out_path: str, time_chunk: int, label: str) -> None:
    """Chunk, write to a temp path, consolidate, then atomically publish to out_path."""
    ds_out = ds_out.sortby("time").chunk({"time": time_chunk})

    encoding = {}
    for name, da in ds_out.data_vars.items():
        chunk_sizes = tuple(min(time_chunk, da.sizes[d]) if d == "time" else da.sizes[d] for d in da.dims)
        encoding[name] = {"chunks": chunk_sizes}

    tmp_path = out_path + ".tmp"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)

    print(f"[{label}] writing {ds_out.sizes['time']} timesteps x {ds_out.sizes.get('z1_l', '-')} levels -> {out_path}")
    ds_out.to_zarr(tmp_path, mode="w", encoding=encoding, consolidated=False)
    zarr.consolidate_metadata(tmp_path)
    if os.path.exists(out_path):  # os.replace can't atomically overwrite a non-empty directory
        shutil.rmtree(out_path)
    os.replace(tmp_path, out_path)
    print(f"[{label}] done")


def process_segment(
    seg_id: str,
    obc_dir: str,
    out_dir: str,
    time_chunk: int,
    overwrite: bool,
    levels: list[float] | None,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None,
    label: str | None,
    noon_offset: bool,
    level_pairs: bool = False,
) -> None:
    direction, long_prefix = SEGMENTS[seg_id]
    suffix = f"_{label}" if label else ""
    out_path = os.path.join(out_dir, f"obc_{direction}{suffix}.zarr")
    if os.path.exists(out_path) and not overwrite:
        print(f"[{direction}] {out_path} already exists, skipping (use --overwrite to redo)")
        return

    src_path = os.path.join(obc_dir, f"forcing_obc_segment_{seg_id}.nc")
    if not os.path.exists(src_path):
        print(f"[{direction}] {src_path} not found, skipping")
        return

    # decode_times=False: we decode 'time' ourselves in _decode_obc_time (see calendar caveat
    # in the module docstring) rather than letting xarray apply the file's declared calendar.
    ds = xr.open_dataset(src_path, decode_times=False, decode_timedelta=False, chunks={"time": time_chunk})
    long_dim = f"{long_prefix}_segment_{seg_id}"
    other_prefix = "ny" if long_prefix == "nx" else "nx"
    singleton_dim = f"{other_prefix}_segment_{seg_id}"  # size-1 cross-boundary dim, dropped below
    target_dim = "xh" if long_prefix == "nx" else "yh"
    # Full flattened coordinate (q,h,q,h,...,q); center points are the odd indices.
    flat_coord = ds[f"lon_segment_{seg_id}" if target_dim == "xh" else f"lat_segment_{seg_id}"]
    center_coord = flat_coord.isel({long_dim: slice(1, None, 2)}).values

    data_vars = {}
    for prefix, canonical in VAR_RENAME.items():
        src_var = f"{prefix}_segment_{seg_id}"
        if src_var not in ds:
            continue
        da = ds[src_var].isel({long_dim: slice(1, None, 2)}).rename({long_dim: target_dim})
        da = da.assign_coords({target_dim: center_coord})
        if singleton_dim in da.dims:
            da = da.squeeze(dim=singleton_dim, drop=True)
        # 3D vars carry their own per-variable level dim (nz_segment_XXX_<prefix>); rename to z1_l.
        level_dim = f"nz_segment_{seg_id}_{prefix}"
        if level_dim in da.dims:
            da = da.rename({level_dim: "z1_l"}).assign_coords(z1_l=ds["depth"].values)
        data_vars[canonical] = da

    # Keep the z1_l-carrying (3D) and flat (SSH) variables in separate Datasets until after
    # level handling: SSH's DataArray never had a z1_l coordinate assigned above, so building
    # it into its own Dataset (rather than merging via a drop_vars split of one combined
    # Dataset) avoids leaving a stale full-length z1_l coordinate attached, which would
    # otherwise make the final xr.merge outer-join on mismatched z1_l labels.
    level_vars = {k: v for k, v in data_vars.items() if "z1_l" in v.dims}
    flat_vars = {k: v for k, v in data_vars.items() if "z1_l" not in v.dims}
    ds_level = xr.Dataset(level_vars)
    if level_pairs:
        ds_level = pairwise_thickness_average(ds_level)
    elif levels is not None:
        ds_level = ds_level.sel(z1_l=levels, method="nearest")
    ds_out = xr.merge([ds_level, xr.Dataset(flat_vars)])
    ds_out = _decode_obc_time(ds_out, noon_offset)
    if date_range is not None:
        ds_out = ds_out.sel(time=slice(*date_range))

    _write_zarr(ds_out, out_path, time_chunk, label=direction)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--obc-dir", default=DEFAULT_OBC_DIR, help="Directory with forcing_obc_segment_00{1..4}.nc.")
    parser.add_argument(
        "--out-dir",
        default=f"/glade/derecho/scratch/{getuser()}/rmom6_regional/preprocessed",
        help="Directory to write obc_<direction>[_<label>].zarr stores into.",
    )
    parser.add_argument("--segments", default="001,002,003,004", help="Comma-separated segment ids to process.")
    parser.add_argument("--start-date", default=None, help="If set (with --end-date), restrict output to this range.")
    parser.add_argument("--end-date", default=None, help="End of the date range (inclusive), used with --start-date.")
    parser.add_argument("--sample-label", default="sample", help="Output filename suffix for a date-limited run.")
    parser.add_argument(
        "--levels",
        default=None,
        help="Comma-separated target depths in metres, nearest-matched to native z1_l layers. "
        "Omit to keep all 50 levels. Mutually exclusive with --level-pairs.",
    )
    parser.add_argument(
        "--level-pairs",
        action="store_true",
        help="Instead of a sparse --levels subset, thickness-weighted-average every adjacent "
        "pair of native z1_l layers (0,1), (2,3), ... down to 25 levels -- see "
        "pairwise_thickness_average(). Mutually exclusive with --levels.",
    )
    parser.add_argument(
        "--time-chunk",
        type=int,
        default=1,
        help="Zarr chunk size along the time dimension. Keep at 1: training reads exactly one "
        "timestep per __getitem__, so a larger time chunk is decompressed in full and then "
        "discarded (best for autoregressive fetching).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Reprocess segments whose output already exists.")
    parser.add_argument(
        "--no-noon-offset",
        dest="noon_offset",
        action="store_false",
        help="Do not shift OBC day-counts by +12h to match the main archive's noon-centered daily means.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the processing plan and exit without doing any I/O."
    )
    parser.set_defaults(noon_offset=True)
    args = parser.parse_args()

    if args.level_pairs and args.levels:
        parser.error("--levels and --level-pairs are mutually exclusive")
    levels = parse_levels(args.levels)
    date_range = None
    label = None
    if args.start_date or args.end_date:
        if not (args.start_date and args.end_date):
            parser.error("--start-date and --end-date must be given together")
        date_range = (pd.Timestamp(args.start_date), pd.Timestamp(args.end_date))
        label = args.sample_label

    segments = [s.strip() for s in args.segments.split(",")]

    print(f"obc dir     : {args.obc_dir}")
    print(f"out dir     : {args.out_dir}")
    print(f"segments    : {[f'{s} ({SEGMENTS[s][0]})' for s in segments]}")
    print(
        f"levels      : {'pairwise thickness-average (50 -> 25)' if args.level_pairs else ('all 50' if levels is None else levels)}"
    )
    print(f"time chunk  : {args.time_chunk}")
    print(f"noon offset : {args.noon_offset}")
    print(
        f"date range  : {'full available range' if date_range is None else f'{date_range[0].date()} .. {date_range[1].date()} (label: {label})'}"
    )
    if args.dry_run:
        print("--dry-run: exiting without processing")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    for seg_id in segments:
        process_segment(
            seg_id,
            args.obc_dir,
            args.out_dir,
            args.time_chunk,
            args.overwrite,
            levels,
            date_range,
            label,
            args.noon_offset,
            level_pairs=args.level_pairs,
        )


if __name__ == "__main__":
    main()
