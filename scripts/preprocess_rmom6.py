#!/usr/bin/env python
"""Preprocess raw carib12 regional MOM6 archive output into CREDIT-ready Zarr stores.

Combines the prognostic ocean state CREDIT trains on — 3D ``uo``, ``vo``, ``thetao``,
``so`` and 2D ``SSH`` — plus a subset of surface forcing, from the raw model archive
into Zarr stores per year, on a single consistent Arakawa A grid (tracer cell centers,
``yh``/``xh``).

Source layout (one file per month, per output stream)::

    <archive-dir>/*.mom6.h.glorys.YYYY-MM.nc   uo(yh,xq) vo(yq,xh) thetao(yh,xh) so(yh,xh)
    <archive-dir>/*.mom6.h.sfc.YYYY-MM.nc      SSH(yh,xh)  (plus other surface diagnostics, unused)
    <archive-dir>/*.mom6.h.atm.YYYY-MM.nc      tauuo(yh,xq) tauvo(yq,xh) hfds(yh,xh) (surface forcing)

``uo``/``tauuo`` live on the west/east cell faces (``xq``, one longer than ``xh``) and
``vo``/``tauvo`` on the north/south cell faces (``yq``, one longer than ``yh``) — MOM6's
Arakawa C grid, output in symmetric-grid form (N+1 face points bracketing N tracer cells).
These are regridded to cell centers with a plain xarray/numpy center-average
(``center[i] = 0.5 * (face[i] + face[i+1])``, same result as the earlier one-off
``messy_convert_to_Arakawa_A_grid.py`` script), staying lazy over dask arrays so a full year
never needs to be held in memory at once. ``hfds``/``friver`` (and other scalar surface
fields) are already on the tracer grid and need no regrid.

No ``xgcm`` dependency: this script previously used ``xgcm.Grid(...).interp(da, axis,
to="center")`` for the regrid (see git history), but that required a separate conda env
(``npl``) with ``xgcm`` installed, which the rest of this pipeline's scripts don't need.
``center_average()`` below reproduces the identical formula by hand for this domain's
symmetric grid (N+1 face points bracketing N tracer cells, first/last are domain edges) —
verified to match the prior xgcm-based output. Run this script in the same env as everything
else (``miles-credit-casper`` / ``credit-derecho``); no env-switching needed anymore. See
``OCEAN_MIGRATION_NOTES.md`` §5 for the parallel note on ``build_rmom6_pointwise_scaler.py``,
which already used this manual approach (that script motivated bringing it here too).

Output is two Zarr stores per year — ``rmom6_prognostic_<year>.zarr`` and
``rmom6_forcing_<year>.zarr`` — so both match ``credit.datasets.local.LocalDataset``'s
default annual ``filename_time_format: "%Y"`` file-mapping with no loader changes
required: point a gen2 config's ``...prognostic.path``/``...dynamic_forcing.path`` at
``<out-dir>/rmom6_prognostic_%Y.zarr`` / ``<out-dir>/rmom6_forcing_%Y.zarr``. The
prognostic store is chunked ``{time: <time-chunk>, z1_l: -1, yh: -1, xh: -1}`` (one chunk
spans every level) since level subsetting already happens here at preprocessing time via
``--levels`` and every training step reads all of a variable's levels for its timestep —
per-level chunking was tried first but multiplies real per-step chunk-read fan-out by the
level count for no benefit any current workflow uses; the forcing store has no level
dimension and is chunked ``{time: <time-chunk>, yh: -1, xh: -1}``.

``--time-chunk`` must stay at its default of 1. The dataset reads exactly one timestep per
``__getitem__`` (``ds.sel(time=t)``), and Zarr can only fetch whole chunks: a time chunk of
*n* decompresses *n* timesteps' worth of every level and then throws away *n-1* of them.
At the ``--time-chunk 30`` this script originally defaulted to, one 25-level timestep cost
**9.0 s** to read (1.04 GB of zstd decompression per 3D variable) versus **0.26 s** at 1 —
which made training ~30x dataloader-bound and dwarfed the GPU work entirely.

Level subsetting (``--levels``) happens here, at preprocessing time, matched to the
nearest native ``z1_l`` layer — matching the "provide a subset in metres, nearest-
matched" convention already documented in ``config/rmom6_regional.yml``. Native
``z1_l`` layer depths (metres, 50 layers) for reference when choosing a subset:

    0.49, 1.54, 2.65, 3.82, 5.08, 6.44, 7.93, 9.57, 11.40, 13.47, 15.81, 18.50,
    21.60, 25.21, 29.44, 34.43, 40.34, 47.37, 55.76, 65.81, 77.85, 92.33, 109.73,
    130.67, 155.85, 186.13, 222.48, 266.04, 318.13, 380.21, 453.94, 541.09, 643.57,
    763.33, 902.34, 1062.44, 1245.29, 1452.25, 1684.28, 1941.89, 2225.08, 2533.34,
    2865.70, 3220.82, 3597.03, 3992.48, 4405.22, 4833.29, 5274.78, 5727.92

``--level-pairs`` is an alternative to ``--levels``: instead of discarding half the native
layers, it thickness-weighted-averages each adjacent pair (0,1), (2,3), ... down to 25 levels,
so every native layer still contributes (see ``pairwise_thickness_average()``). The archive
carries only layer-center depths, not thicknesses, so thickness is reconstructed from ``z1_l``
itself (``_layer_interfaces()``) by exploiting that a center is the midpoint of its two
bracketing interfaces.

Forcing variables currently produced (``FORCING_SPEC``) — canonical name -> archive
variable, from ``*.atm.YYYY-MM.nc``:

    taux             -> tauuo   (regridded from xq)
    tauy             -> tauvo   (regridded from yq)
    net_heat_surface -> hfds    (already tracer-grid)
    runoff           -> friver  (already tracer-grid)

``FORCING_SPEC`` is a plain dict — add an entry (``source_var``, and ``axis`` ("X"/"Y") if
it needs the C-to-A regrid, else ``None``) to produce more forcing variables later, e.g.
``p_surf -> pso`` (no regrid needed). ``pso``/``wfo``/``evs`` are also present in the archive's
``*.atm.YYYY-MM.nc`` files but deliberately not added here yet — only ``runoff`` was requested
so far. Pass ``--skip-forcing`` to only produce the prognostic store.

Left untouched by this script (out of scope — see OCEAN_MIGRATION_NOTES.md):
    - open boundary conditions (a separate source grid entirely; not in this archive)
    - static geometry (wet, deptho, ...)
    - normalization statistics / scaler generation (see scripts/build_rmom6_scaler.py)

Usage::

    # Preview what would run without touching any data
    python scripts/preprocess_rmom6.py --dry-run

    # Full 20-year archive, all 50 levels, default output location
    python scripts/preprocess_rmom6.py

    # Single year, upper-ocean subset (nearest-matched), for a quick test run
    python scripts/preprocess_rmom6.py --years 2000 --levels 0.5,5,15,30,60,100

    # Single year, pairwise thickness-weighted levels (50 -> 25), for a quick test run
    python scripts/preprocess_rmom6.py --years 2000 --level-pairs

    # Prognostic state only, skip forcing
    python scripts/preprocess_rmom6.py --skip-forcing

    # Quick 15-day sample (single month's files only, fast) -> rmom6_prognostic_sample.zarr
    # / rmom6_forcing_sample.zarr, not tied to the annual %Y naming (see --start-date).
    python scripts/preprocess_rmom6.py --start-date 2000-01-01 --end-date 2000-01-15

Dependencies: xarray, dask, zarr, numpy, pandas.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
from getpass import getuser

import numpy as np
import pandas as pd
import xarray as xr
import zarr

DEFAULT_ARCHIVE_DIR = (
    "/glade/derecho/scratch/ajanney/archive/"
    "carib12_runoff_tides_rmax600_f200_gioGlofas_gioNNSM/ocn/hist"
)
GLORYS_VARS = ["uo", "vo", "thetao", "so"]
SFC_VARS = ["SSH"]
# canonical name -> {"source_var": archive var name, "axis": "X"/"Y"/None}
# axis is the center_average() axis to regrid the face variable onto the tracer center along
# ("X" like uo/tauuo on xq, "Y" like vo/tauvo on yq), or None for variables already on the
# tracer grid (yh, xh). Add entries here to produce more forcing variables (e.g.
# "p_surf": {"source_var": "pso", "axis": None}).
FORCING_SPEC: dict[str, dict[str, str | None]] = {
    "taux": {"source_var": "tauuo", "axis": "X"},
    "tauy": {"source_var": "tauvo", "axis": "Y"},
    "net_heat_surface": {"source_var": "hfds", "axis": None},
    "runoff": {"source_var": "friver", "axis": None},
}


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


def parse_levels(spec: str | None) -> list[float] | None:
    """Parse a comma-separated list of target depths (metres) into floats, or None for all levels."""
    if spec is None:
        return None
    return [float(v) for v in spec.split(",")]


def _glob_stream_files(archive_dir: str, stream: str, year: int, months: list[int] | None) -> list[str]:
    """Glob one archive stream's monthly files for *year*, optionally restricted to *months*."""
    if months is None:
        return sorted(glob.glob(os.path.join(archive_dir, f"*.mom6.h.{stream}.{year}-*.nc")))
    files = []
    for m in months:
        files.extend(glob.glob(os.path.join(archive_dir, f"*.mom6.h.{stream}.{year}-{m:02d}.nc")))
    return sorted(files)


def _date_range_months(date_range: tuple[pd.Timestamp, pd.Timestamp] | None) -> tuple[int | None, list[int] | None]:
    """Return (year, months) implied by a (start, end) date range, or (None, None) if unset.

    Raises ValueError if the range spans more than one calendar year — sample runs are
    written under a single year's monthly files (see docstring re: rmom6_*_sample.zarr).
    """
    if date_range is None:
        return None, None
    start, end = date_range
    if start.year != end.year:
        raise ValueError(f"--start-date/--end-date must fall within a single year; got {start.date()}..{end.date()}")
    return start.year, list(range(start.month, end.month + 1))


def center_average(da: xr.DataArray, axis: str) -> xr.DataArray:
    """Average an Arakawa-C face variable onto tracer cell centers (Arakawa A), no xgcm.

    MOM6 symmetric output carries xq/yq as N+1 face points bracketing N tracer cells
    (including both domain edges): ``center[i] = 0.5 * (face[i] + face[i+1])``. Stays lazy
    under dask (plain xarray isel + arithmetic, no compute triggered).

    Args:
        da: DataArray on a face grid (e.g. uo on 'xq', vo on 'yq').
        axis: 'X' for xq-faced vars (uo/tauuo), 'Y' for yq-faced vars (vo/tauvo).

    Returns:
        DataArray on the corresponding tracer-center dimension (xh or yh), renamed from the
        face dim with no coordinate of its own -- the caller combines it into a Dataset
        alongside a tracer-grid variable (e.g. thetao/hfds) that already carries the real
        xh/yh coordinate values, which xarray then adopts for the shared dimension.
    """
    face_dim = "xq" if axis == "X" else "yq"
    center_dim = "xh" if axis == "X" else "yh"
    left = da.isel({face_dim: slice(0, -1)}).drop_vars(face_dim, errors="ignore")
    right = da.isel({face_dim: slice(1, None)}).drop_vars(face_dim, errors="ignore")
    return (0.5 * (left + right)).rename({face_dim: center_dim})


def _layer_interfaces(z1_l: np.ndarray) -> np.ndarray:
    """Reconstruct MOM6 layer interface depths from cell-center depths (z1_l).

    The archive carries only layer centers, not interfaces or thicknesses. A center is by
    definition the midpoint of its two bracketing interfaces, so interfaces can be recovered
    top-down: ``z1_i[0] = 0`` (surface), ``z1_i[k+1] = 2 * z1_l[k] - z1_i[k]``. Thickness is
    then the gap between successive interfaces (see ``pairwise_thickness_average()``).
    """
    z1_i = np.zeros(len(z1_l) + 1)
    for k, zc in enumerate(z1_l):
        z1_i[k + 1] = 2.0 * zc - z1_i[k]
    return z1_i


def pairwise_thickness_average(ds: xr.Dataset) -> xr.Dataset:
    """Merge adjacent pairs of z1_l levels into one, thickness-weighted: 50 levels -> 25.

    An alternative to ``--levels``' sparse nearest-match subsetting, which simply discards half
    the native layers. Every adjacent pair (0,1), (2,3), ... is combined into one output layer::

        T' = (T_i * dz_i + T_{i+1} * dz_{i+1}) / (dz_i + dz_{i+1})

    where ``dz`` is each native layer's thickness from ``_layer_interfaces()``. This keeps every
    native layer's contribution (at the cost of some vertical smoothing within each merged
    pair) instead of dropping layers outright. The merged layer's ``z1_l`` coordinate is the
    same thickness-weighted average applied to depth itself -- its center of mass.
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


def _fix_mislabeled_calendar(ds: xr.Dataset) -> xr.Dataset:
    """Override the archive's mislabeled 'gregorian' time calendar attribute.

    The carib12 archive's monthly files declare ``calendar: gregorian`` (units ``days since
    0001-01-01``), but the stored values are actually ``proleptic_gregorian`` -- confirmed by
    direct inspection: file ``...glorys.2003-01.nc``'s raw time values decode to
    2002-12-30..2003-01-29 under 'gregorian' (a spurious ~2-day-early shift that grows with
    distance from the 1582 Julian/Gregorian reform date, since 'gregorian'/'standard' honors
    that historical transition and 'proleptic_gregorian' does not) but to the *correct*
    2003-01-01..2003-01-31 under 'proleptic_gregorian', exactly matching the file's own name.
    This is a known MOM6/CESM archive mislabeling, not a real gap in the data -- discovered via
    a training-time KeyError at year boundaries (each calendar-year zarr store was silently
    missing its own last ~2 days under the wrong calendar). Called as `open_mfdataset`'s
    ``preprocess`` hook, applied per-file before combining/decoding.
    """
    if "time" in ds.coords and ds.time.attrs.get("calendar") == "gregorian":
        ds = ds.copy()
        ds.time.attrs["calendar"] = "proleptic_gregorian"
    return ds


def open_month_files(files: list[str], varnames: list[str], time_chunk: int) -> xr.Dataset:
    chunks = {"time": time_chunk}
    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        decode_times=False,  # calendar corrected below, then decoded explicitly
        decode_timedelta=False,
        preprocess=_fix_mislabeled_calendar,
        chunks=chunks,
    )
    ds = xr.decode_cf(ds, decode_timedelta=False)
    return ds[varnames]


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
    # LocalDataset reopens the store fresh on every __getitem__ call, so consolidated
    # metadata (one read instead of a listing per array) matters for training throughput.
    zarr.consolidate_metadata(tmp_path)
    if os.path.exists(out_path):  # os.replace can't atomically overwrite a non-empty directory
        shutil.rmtree(out_path)
    os.replace(tmp_path, out_path)
    print(f"[{label}] done")


def process_prognostic_year(
    year: int,
    archive_dir: str,
    out_dir: str,
    levels: list[float] | None,
    time_chunk: int,
    overwrite: bool,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    months: list[int] | None = None,
    label: str | None = None,
    level_pairs: bool = False,
) -> None:
    label = label or str(year)
    out_path = os.path.join(out_dir, f"rmom6_prognostic_{label}.zarr")
    if os.path.exists(out_path) and not overwrite:
        print(f"[{label}] {out_path} already exists, skipping (use --overwrite to redo)")
        return

    glorys_files = _glob_stream_files(archive_dir, "glorys", year, months)
    sfc_files = _glob_stream_files(archive_dir, "sfc", year, months)
    if not glorys_files or not sfc_files:
        print(f"[{label}] no glorys/sfc files found under {archive_dir}, skipping")
        return
    expected = months and len(months) or 12
    if len(glorys_files) != expected or len(sfc_files) != expected:
        print(f"[{label}] WARNING: expected {expected} monthly file(s) each, found {len(glorys_files)} glorys / {len(sfc_files)} sfc")

    ds_g = open_month_files(glorys_files, GLORYS_VARS, time_chunk)
    ds_s = open_month_files(sfc_files, SFC_VARS, time_chunk)

    if level_pairs:
        ds_g = pairwise_thickness_average(ds_g)
    elif levels is not None:
        ds_g = ds_g.sel(z1_l=levels, method="nearest")
    if date_range is not None:
        ds_g = ds_g.sel(time=slice(*date_range))
        ds_s = ds_s.sel(time=slice(*date_range))

    n_glorys, n_sfc = ds_g.sizes["time"], ds_s.sizes["time"]
    ds_g, ds_s = xr.align(ds_g, ds_s, join="inner")
    if ds_g.sizes["time"] != n_glorys or ds_s.sizes["time"] != n_sfc:
        print(
            f"[{label}] WARNING: glorys/sfc time mismatch — using "
            f"{ds_g.sizes['time']} of {n_glorys} glorys / {n_sfc} sfc timesteps"
        )

    uo_c = center_average(ds_g["uo"], "X")
    vo_c = center_average(ds_g["vo"], "Y")

    ds_out = xr.Dataset(
        {
            "uo": uo_c,
            "vo": vo_c,
            "thetao": ds_g["thetao"],
            "so": ds_g["so"],
            "SSH": ds_s["SSH"],
        }
    )
    _write_zarr(ds_out, out_path, time_chunk, label=f"{label} prognostic")


def process_forcing_year(
    year: int,
    archive_dir: str,
    out_dir: str,
    time_chunk: int,
    overwrite: bool,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    months: list[int] | None = None,
    label: str | None = None,
) -> None:
    label = label or str(year)
    out_path = os.path.join(out_dir, f"rmom6_forcing_{label}.zarr")
    if os.path.exists(out_path) and not overwrite:
        print(f"[{label}] {out_path} already exists, skipping (use --overwrite to redo)")
        return

    atm_files = _glob_stream_files(archive_dir, "atm", year, months)
    if not atm_files:
        print(f"[{label}] no atm files found under {archive_dir}, skipping forcing")
        return
    expected = months and len(months) or 12
    if len(atm_files) != expected:
        print(f"[{label}] WARNING: expected {expected} monthly atm file(s), found {len(atm_files)}")

    source_vars = [spec["source_var"] for spec in FORCING_SPEC.values()]
    ds_a = open_month_files(atm_files, source_vars, time_chunk)
    if date_range is not None:
        ds_a = ds_a.sel(time=slice(*date_range))

    data_vars = {}
    for canonical, spec in FORCING_SPEC.items():
        da = ds_a[spec["source_var"]]
        axis = spec["axis"]
        data_vars[canonical] = da if axis is None else center_average(da, axis)

    ds_out = xr.Dataset(data_vars)
    _write_zarr(ds_out, out_path, time_chunk, label=f"{label} forcing")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--archive-dir", default=DEFAULT_ARCHIVE_DIR, help="Directory of raw carib12 archive files.")
    parser.add_argument(
        "--out-dir",
        default=f"/glade/derecho/scratch/{getuser()}/rmom6_regional/preprocessed",
        help="Directory to write rmom6_prognostic_<year>.zarr stores into.",
    )
    parser.add_argument("--years", default="2000-2019", help="Years to process, e.g. '2000-2019', '2003', '2001,2005'.")
    parser.add_argument(
        "--start-date",
        default=None,
        help="If set (with --end-date), process only this date range (must fall within one calendar "
        "year) and write to rmom6_*_sample.zarr instead of the annual rmom6_*_<year>.zarr naming. "
        "Overrides --years. Only the needed month(s) of source files are opened.",
    )
    parser.add_argument("--end-date", default=None, help="End of the date range (inclusive), used with --start-date.")
    parser.add_argument(
        "--sample-label",
        default="sample",
        help="Output filename suffix for a --start-date/--end-date run, i.e. rmom6_prognostic_<label>.zarr.",
    )
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
        "discarded (a 30-timestep chunk cost 9.0 s/timestep vs 0.26 s at 1).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Reprocess years whose output already exists.")
    parser.add_argument(
        "--skip-forcing", action="store_true", help="Only produce the prognostic store; skip rmom6_forcing_<year>.zarr."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the processing plan and exit without doing any I/O.")
    args = parser.parse_args()

    if args.level_pairs and args.levels:
        parser.error("--levels and --level-pairs are mutually exclusive")
    levels = parse_levels(args.levels)
    date_range = None
    if args.start_date or args.end_date:
        if not (args.start_date and args.end_date):
            parser.error("--start-date and --end-date must be given together")
        date_range = (pd.Timestamp(args.start_date), pd.Timestamp(args.end_date))

    print(f"archive dir : {args.archive_dir}")
    print(f"out dir     : {args.out_dir}")
    print(f"levels      : {'pairwise thickness-average (50 -> 25)' if args.level_pairs else ('all 50' if levels is None else levels)}")
    print(f"time chunk  : {args.time_chunk}")
    print(f"forcing     : {'skipped' if args.skip_forcing else ', '.join(FORCING_SPEC)}")

    if date_range is not None:
        year, months = _date_range_months(date_range)
        print(f"date range  : {date_range[0].date()} .. {date_range[1].date()}  (label: {args.sample_label})")
        if args.dry_run:
            print("--dry-run: exiting without processing")
            return
        os.makedirs(args.out_dir, exist_ok=True)
        process_prognostic_year(
            year, args.archive_dir, args.out_dir, levels, args.time_chunk, args.overwrite,
            date_range=date_range, months=months, label=args.sample_label, level_pairs=args.level_pairs,
        )
        if not args.skip_forcing:
            process_forcing_year(
                year, args.archive_dir, args.out_dir, args.time_chunk, args.overwrite,
                date_range=date_range, months=months, label=args.sample_label,
            )
        return

    years = parse_years(args.years)
    print(f"years       : {years[0]}-{years[-1]} ({len(years)} years)")
    if args.dry_run:
        print("--dry-run: exiting without processing")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    for year in years:
        process_prognostic_year(
            year, args.archive_dir, args.out_dir, levels, args.time_chunk, args.overwrite,
            level_pairs=args.level_pairs,
        )
        if not args.skip_forcing:
            process_forcing_year(year, args.archive_dir, args.out_dir, args.time_chunk, args.overwrite)


if __name__ == "__main__":
    main()
