#!/usr/bin/env python
"""Animate rMOM6 rollout surface fields as GIFs — emulator vs truth vs error, over lead time.

Two modes:

``--rollout-dir`` (per model)
    One GIF per surface field, 3 panels: emulator | truth | emulator-truth. Fields are
    SST (thetao level 0), SSS (so level 0), SSH, and surface current speed
    sqrt(uo^2+vo^2) at level 0.

``--compare`` (across models)
    One GIF per field laying every model out on a grid next to the truth, so a shared
    failure mode (e.g. a bathymetry-shaped imprint) is obvious at a glance.

Alignment follows eval_rmom6_rollout.py: emulator and truth are both daily, and emulator
forecast step k corresponds to truth day index k, so the truth is sliced from index 1.

Land: the emulator writes filled values over land, the truth writes NaN, so every panel is
masked by the truth's NaN pattern — otherwise the fill would dominate the color scale.

Color limits are computed once from the truth over the whole window and held fixed across
frames. That is deliberate: if the emulator drifts, it saturates the colorbar rather than
silently rescaling and looking stable.

Usage::

    python scripts/animate_rmom6_rollout.py \\
        --rollout-dir /glade/.../rollout_sanity_60d --label levelwise_singlestep_notendency \\
        --out /glade/.../eval_rmom6/gifs

    python scripts/animate_rmom6_rollout.py --compare \\
        --runs-root /glade/derecho/scratch/$USER/CREDIT_runs \\
        --subdir rollout_sanity_60d --out /glade/.../eval_rmom6/gifs

Dependencies: xarray, numpy, matplotlib (Pillow writer), dask.
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

TRUTH_TMPL = "/glade/derecho/scratch/ajanney/rmom6_regional/preprocessed/rmom6_prognostic_{year}.zarr"

# field -> (label, unit, colormap)
FIELDS = {
    "SST": ("SST (thetao, level 0)", "°C", "RdYlBu_r"),
    "SSS": ("SSS (so, level 0)", "PSU", "viridis"),
    "SSH": ("SSH", "m", "RdBu_r"),
    "SPD": ("surface speed |u|", "m s$^{-1}$", "magma"),
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_emulator(rollout_dir: str) -> xr.Dataset:
    subdirs = sorted(glob.glob(os.path.join(rollout_dir, "*Z")))
    if not subdirs:
        raise FileNotFoundError(f"no init-time subdir (*Z) under {rollout_dir}")
    files = sorted(glob.glob(os.path.join(subdirs[0], "*.nc")))
    if not files:
        raise FileNotFoundError(f"no .nc files under {subdirs[0]}")
    return xr.open_mfdataset(files, combine="by_coords").sortby("time")


def surface_fields(ds: xr.Dataset, n_steps: int, level_dim: str | None = None) -> dict[str, np.ndarray]:
    """Pull the four surface fields as (time, y, x) float32 arrays.

    Selecting level 0 *before* .values is what keeps this cheap: the full 25-level record for
    four variables over 60 steps is ~40 GB, the surface slice is ~330 MB.
    """
    if level_dim is None:
        # The writer's level coord name varies (z1_l, level, or a bare index fallback), so
        # find it rather than hardcoding: it is the dim that is neither time nor horizontal.
        dims = [d for d in ds["thetao"].dims if d not in ("time", "yh", "xh", "y", "x", "latitude", "longitude")]
        if len(dims) != 1:
            raise ValueError(f"could not identify the level dim of thetao among {ds['thetao'].dims}")
        level_dim = dims[0]

    sel = dict(time=slice(0, n_steps))
    top = {v: ds[v].isel(**sel).isel({level_dim: 0}) for v in ("thetao", "so", "uo", "vo")}
    ssh = ds["SSH"].isel(**sel)
    if level_dim in ssh.dims:  # some writers keep a singleton level on 2D vars
        ssh = ssh.isel({level_dim: 0})

    out = {
        "SST": np.asarray(top["thetao"].values, np.float32),
        "SSS": np.asarray(top["so"].values, np.float32),
        "SSH": np.asarray(ssh.values, np.float32),
    }
    u = np.asarray(top["uo"].values, np.float32)
    v = np.asarray(top["vo"].values, np.float32)
    out["SPD"] = np.sqrt(u * u + v * v)
    return out


def load_truth_surface(n_steps: int, years=(2018, 2019)) -> dict[str, np.ndarray]:
    paths = [TRUTH_TMPL.format(year=y) for y in years]
    truth = xr.open_mfdataset(paths, engine="zarr", combine="by_coords")
    # Emulator step k == truth day index k, so drop the IC day.
    return surface_fields(truth.isel(time=slice(1, n_steps + 1)), n_steps)


def apply_land_mask(emul: dict, truth: dict) -> None:
    """Crop any OBC halo, then blank the emulator wherever the truth is land. In place.

    The ``ocean_obc_halo`` preblock appends a prescribed one-cell boundary (last row, last
    column) to the model grid, so rollouts written after that change are 458x760 rather than
    457x759. Those cells are boundary condition rather than prediction and have no counterpart
    in the truth zarr, so they are dropped before any comparison. Cropping to the truth's shape
    rather than a hardcoded size keeps this working for pre-halo rollouts too.
    """
    for k in emul:
        th, tw = truth[k].shape[-2:]
        if emul[k].shape[-2:] != (th, tw):
            emul[k] = emul[k][..., :th, :tw]
        emul[k] = np.where(np.isfinite(truth[k]), emul[k], np.nan)


# ---------------------------------------------------------------------------
# Color limits
# ---------------------------------------------------------------------------
def field_limits(truth_arr: np.ndarray) -> tuple[float, float]:
    finite = truth_arr[np.isfinite(truth_arr)]
    return float(np.percentile(finite, 1)), float(np.percentile(finite, 99))


def diff_limit(emul_arr: np.ndarray, truth_arr: np.ndarray) -> float:
    d = emul_arr - truth_arr
    finite = np.abs(d[np.isfinite(d)])
    if finite.size == 0:
        return 1.0
    return max(float(np.percentile(finite, 99)), 1e-6)


# ---------------------------------------------------------------------------
# Per-model GIFs
# ---------------------------------------------------------------------------
def animate_model(emul: dict, truth: dict, label: str, out_dir: str, fps: int, dpi: int) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    n_steps = next(iter(emul.values())).shape[0]
    written = []

    for key, (title, unit, cmap) in FIELDS.items():
        e, t = emul[key], truth[key]
        vmin, vmax = field_limits(t)
        dlim = diff_limit(e, t)

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)
        ims = []
        for ax, (arr, cm, lo, hi, sub) in zip(
            axes,
            [
                (e, cmap, vmin, vmax, "emulator"),
                (t, cmap, vmin, vmax, "truth (MOM6)"),
                (e - t, "RdBu_r", -dlim, dlim, "emulator − truth"),
            ],
        ):
            im = ax.imshow(arr[0], origin="lower", cmap=cm, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_title(sub, fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.85, label=unit)
            ims.append(im)
        sup = fig.suptitle("", fontsize=13)

        def update(f, ims=ims, e=e, t=t, sup=sup, title=title):
            ims[0].set_data(e[f])
            ims[1].set_data(t[f])
            ims[2].set_data(e[f] - t[f])
            sup.set_text(f"{title} — {label} — lead day {f + 1}/{n_steps}")
            return ims

        path = os.path.join(out_dir, f"{label}_{key}.gif")
        FuncAnimation(fig, update, frames=n_steps, blit=False).save(path, writer=PillowWriter(fps=fps), dpi=dpi)
        plt.close(fig)
        written.append(path)
        print(f"  wrote {path}")
    return written


# ---------------------------------------------------------------------------
# Cross-model comparison GIFs
# ---------------------------------------------------------------------------
def animate_compare(per_model: dict[str, dict], truth: dict, out_dir: str, fps: int, dpi: int) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    labels = sorted(per_model)
    n_steps = min(next(iter(m.values())).shape[0] for m in per_model.values())
    n_panels = len(labels) + 1
    ncol = 3
    nrow = int(np.ceil(n_panels / ncol))
    written = []

    for key, (title, unit, cmap) in FIELDS.items():
        vmin, vmax = field_limits(truth[key])
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.4 * nrow), constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()
        ims = []

        im = axes[0].imshow(truth[key][0], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        axes[0].set_title("truth (MOM6)", fontsize=10, fontweight="bold")
        ims.append((im, None))
        for ax, lab in zip(axes[1:], labels):
            arr = per_model[lab][key]
            im = ax.imshow(arr[0], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(lab.replace("_", " "), fontsize=9)
            ims.append((im, lab))
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axes[n_panels:]:
            ax.axis("off")
        fig.colorbar(ims[0][0], ax=axes[:n_panels], shrink=0.6, label=unit)
        sup = fig.suptitle("", fontsize=14)

        def update(f, ims=ims, key=key, sup=sup, title=title):
            for im, lab in ims:
                im.set_data(truth[key][f] if lab is None else per_model[lab][key][f])
            sup.set_text(f"{title} — all models on the truth's color scale — lead day {f + 1}/{n_steps}")
            return [im for im, _ in ims]

        path = os.path.join(out_dir, f"COMPARE_{key}.gif")
        FuncAnimation(fig, update, frames=n_steps, blit=False).save(path, writer=PillowWriter(fps=fps), dpi=dpi)
        plt.close(fig)
        written.append(path)
        print(f"  wrote {path}")
    return written


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollout-dir", help="A single rollout output dir (contains an <init>Z/ subdir).")
    ap.add_argument("--label", help="Name for the single-model mode.")
    ap.add_argument("--compare", action="store_true", help="Build cross-model comparison GIFs.")
    ap.add_argument(
        "--runs-root",
        default=f"/glade/derecho/scratch/{os.environ.get('USER', 'ajanney')}/CREDIT_runs",
        help="Root holding the rmom6_regional_*_full run dirs (--compare mode).",
    )
    ap.add_argument("--subdir", default="rollout_sanity_60d", help="Rollout subdir inside each run dir.")
    ap.add_argument("--out", required=True, help="Output directory for the GIFs.")
    ap.add_argument("--steps", type=int, default=None, help="Limit to the first N lead days.")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--dpi", type=int, default=90)
    args = ap.parse_args()

    if args.compare:
        run_dirs = sorted(glob.glob(os.path.join(args.runs_root, "rmom6_regional_*_full")))
        per_model, n_steps = {}, args.steps
        found = []
        for rd in run_dirs:
            roll = os.path.join(rd, args.subdir)
            if not glob.glob(os.path.join(roll, "*Z")):
                print(f"skip (no output yet): {os.path.basename(rd)}")
                continue
            found.append((os.path.basename(rd).replace("rmom6_regional_", "").replace("_full", ""), roll))
        if not found:
            raise SystemExit(f"no rollouts found under {args.runs_root}/*/{args.subdir}")

        for label, roll in found:
            ds = load_emulator(roll)
            steps = ds.sizes["time"] if n_steps is None else min(n_steps, ds.sizes["time"])
            n_steps = steps if n_steps is None else min(n_steps, steps)
            per_model[label] = surface_fields(ds, steps)
            print(f"loaded {label}: {steps} steps")

        # Re-trim to the common length in case models produced different step counts.
        for label in per_model:
            per_model[label] = {k: v[:n_steps] for k, v in per_model[label].items()}
        truth = load_truth_surface(n_steps)
        for label in per_model:
            apply_land_mask(per_model[label], truth)

        print(f"\nbuilding comparison GIFs over {n_steps} lead days for {len(per_model)} models")
        animate_compare(per_model, truth, args.out, args.fps, args.dpi)
    else:
        if not args.rollout_dir or not args.label:
            raise SystemExit("--rollout-dir and --label are required unless --compare is given")
        ds = load_emulator(args.rollout_dir)
        n_steps = ds.sizes["time"] if args.steps is None else min(args.steps, ds.sizes["time"])
        emul = surface_fields(ds, n_steps)
        truth = load_truth_surface(n_steps)
        apply_land_mask(emul, truth)
        print(f"building GIFs for {args.label} over {n_steps} lead days")
        animate_model(emul, truth, args.label, args.out, args.fps, args.dpi)


if __name__ == "__main__":
    main()
