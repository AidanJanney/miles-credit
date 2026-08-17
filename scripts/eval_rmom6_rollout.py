"""
eval_rmom6_rollout.py
=====================
Evaluate a CREDIT gen2 rMOM6 ocean-emulator rollout against the preprocessed
25-level MOM6 truth (the model's actual training target).

Single pass over time (one read per step) accumulates everything:
  1. RMSE vs lead time for SST (thetao l0), SSH, SSS (so l0)   [ocean-masked]
  2. RMSE across depth  (level-wise RMSE, time-averaged)        [profile]
  3. Level-wise RMSE vs lead time                               [heatmap]
  4. Mean surface fields (time-mean maps, emul vs truth)
  5. Spatial surface snapshots at selected lead times
  6. Surface KE (U/V) power spectra, emul vs truth  (thesis eval_wavenumber_2)

Emulator and truth are aligned POSITIONALLY: both are daily starting
2018-01-01T12:00; emulator forecast step k (k>=1) == truth day index k.

Usage:
  python scripts/eval_rmom6_rollout.py \
      --rollout-dir /glade/.../rollout_2018_2yr --label <name> --out /glade/.../eval/<name>
"""

from __future__ import annotations
import argparse
import glob
import os
import sys

import numpy as np
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/glade/work/ajanney/Regional_Ocean_Emulation/archive/Thesis_Archive")
from eval_wavenumber_2 import (  # noqa: E402
    estimate_grid_spacing_km,
    compute_2d_fft_normalized,
    azimuthal_average,
)

TRUTH_TMPL = "/glade/derecho/scratch/ajanney/rmom6_regional/preprocessed/rmom6_prognostic_{year}.zarr"
STATIC = (
    "/glade/derecho/scratch/ajanney/archive/carib12_runoff_tides_rmax600_f200_gioGlofas_gioNNSM/"
    "ocn/hist/carib12_runoff_tides_rmax600_f200_gioGlofas_gioNNSM.mom6.h.static.nc"
)
HORIZONS = [30, 60, 120, 365, 729]
VARS3D = ["thetao", "so", "uo", "vo"]


def load_emulator(rollout_dir, truth_shape=None):
    """Open a rollout, optionally cropping the OBC halo to the truth grid.

    The ``ocean_obc_halo`` preblock appends a prescribed one-cell boundary (last row, last
    column), so rollouts written after that change are 458x760 rather than 457x759. Those cells
    are boundary condition rather than prediction and have no counterpart in the truth zarr.
    Cropping against the truth's own (yh, xh) sizes rather than a hardcoded shape keeps this
    correct for pre-halo rollouts, which need no crop at all.
    """
    subdirs = sorted(glob.glob(os.path.join(rollout_dir, "*Z")))
    if not subdirs:
        raise FileNotFoundError(f"No init-time subdir (*Z) under {rollout_dir}")
    files = sorted(glob.glob(os.path.join(subdirs[0], "*.nc")))
    ds = xr.open_mfdataset(files, combine="by_coords").sortby("time")
    if truth_shape is not None:
        ny, nx = truth_shape
        ydim = "yh" if "yh" in ds.sizes else [d for d in ds.dims if ds.sizes[d] in (ny, ny + 1)][0]
        xdim = "xh" if "xh" in ds.sizes else [d for d in ds.dims if ds.sizes[d] in (nx, nx + 1)][0]
        if ds.sizes[ydim] != ny or ds.sizes[xdim] != nx:
            print(f"cropping OBC halo: ({ds.sizes[ydim]}, {ds.sizes[xdim]}) -> ({ny}, {nx})")
            ds = ds.isel({ydim: slice(0, ny), xdim: slice(0, nx)})
    return ds


def load_truth(n_steps):
    truth = xr.open_mfdataset([TRUTH_TMPL.format(year=y) for y in (2018, 2019)], engine="zarr", combine="by_coords")
    return truth.isel(time=slice(1, n_steps + 1))


def _geo():
    s = xr.open_dataset(STATIC)
    return s["geolon"].values, s["geolat"].values, s["wet"].values


def masked_rmse(pred, true):
    m = np.isfinite(true) & np.isfinite(pred)
    if not m.any():
        return np.nan
    d = pred[m] - true[m]
    return float(np.sqrt(np.mean(d * d)))


# ---------------------------------------------------------------------------
# Single pass: read each step once, accumulate all metrics
# ---------------------------------------------------------------------------
def gather(emul, truth, n_steps, nlev):
    surf = {"SST": np.full(n_steps, np.nan), "SSS": np.full(n_steps, np.nan), "SSH": np.full(n_steps, np.nan)}
    lw = {v: np.full((nlev, n_steps), np.nan) for v in VARS3D}
    # running sums for time-mean surface fields
    msum = {k: None for k in ("SST", "SSS", "SSH")}
    tsum = {k: None for k in ("SST", "SSS", "SSH")}
    snaps, spec = {}, {}
    hset = {h for h in HORIZONS if h <= n_steps}

    for s in range(n_steps):
        E = {v: np.asarray(emul[v].isel(time=s).values, np.float32) for v in VARS3D}
        Essh = np.asarray(emul["SSH"].isel(time=s).values, np.float32)
        T = {v: np.asarray(truth[v].isel(time=s).values, np.float32) for v in VARS3D}
        Tssh = np.asarray(truth["SSH"].isel(time=s).values, np.float32)

        surf["SST"][s] = masked_rmse(E["thetao"][0], T["thetao"][0])
        surf["SSS"][s] = masked_rmse(E["so"][0], T["so"][0])
        surf["SSH"][s] = masked_rmse(Essh, Tssh)
        for v in VARS3D:  # vectorized level-wise RMSE (all nlev at once)
            ev, tv = E[v], T[v]
            m = np.isfinite(ev) & np.isfinite(tv)  # (nlev,H,W)
            d2 = np.where(m, (ev - tv) ** 2, 0.0).reshape(nlev, -1)
            cnt = m.reshape(nlev, -1).sum(1)
            ssum = d2.sum(1)
            lw[v][:, s] = np.where(cnt > 0, np.sqrt(ssum / np.maximum(cnt, 1)), np.nan)

        for key, (ef, tf) in {
            "SST": (E["thetao"][0], T["thetao"][0]),
            "SSS": (E["so"][0], T["so"][0]),
            "SSH": (Essh, Tssh),
        }.items():
            ef = np.where(np.isfinite(tf), ef, np.nan)
            msum[key] = ef if msum[key] is None else np.nansum([msum[key], ef], axis=0)
            tsum[key] = tf if tsum[key] is None else np.nansum([tsum[key], tf], axis=0)

        if (s + 1) in hset:
            snaps[s + 1] = (E["thetao"][0].copy(), T["thetao"][0].copy())
            spec[s + 1] = (E["uo"][0].copy(), E["vo"][0].copy(), T["uo"][0].copy(), T["vo"][0].copy())

    mean_e = {k: msum[k] / n_steps for k in msum}
    mean_t = {k: tsum[k] / n_steps for k in tsum}
    return dict(surf=surf, lw=lw, mean_e=mean_e, mean_t=mean_t, snaps=snaps, spec=spec)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def p_rmse_curves(surf, n_steps, label, out):
    days = np.arange(1, n_steps + 1)
    units = {"SST": "°C", "SSS": "PSU", "SSH": "m"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, k in zip(axes, ("SST", "SSS", "SSH")):
        ax.plot(days, surf[k], lw=1.4)
        for h in HORIZONS:
            if h <= n_steps:
                ax.axvline(h, color="gray", ls=":", lw=0.6, alpha=0.5)
        ax.set_xlabel("lead time [days]")
        ax.set_ylabel(f"RMSE [{units[k]}]")
        ax.set_title(k)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Surface RMSE vs lead time — {label}", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{out}/rmse_surface_vs_leadtime.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def p_depth(lw, levels, label, out):
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    for ax, v in zip(axes, VARS3D):
        prof = np.nanmean(lw[v], axis=1)
        ax.plot(prof, levels, lw=1.5, marker="o", ms=3)
        ax.invert_yaxis()
        ax.set_xlabel(f"RMSE {v}")
        ax.grid(alpha=0.3)
        ax.set_title(v)
    axes[0].set_ylabel("depth [m]")
    fig.suptitle(f"Time-averaged RMSE vs depth — {label}", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{out}/rmse_vs_depth.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def p_heatmap(lw, levels, n_steps, label, out):
    days = np.arange(1, n_steps + 1)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, v in zip(axes, VARS3D):
        im = ax.pcolormesh(days, levels, lw[v], shading="auto", cmap="viridis")
        ax.invert_yaxis()
        ax.set_xlabel("lead time [days]")
        ax.set_title(v)
        fig.colorbar(im, ax=ax, shrink=0.8)
    axes[0].set_ylabel("depth [m]")
    fig.suptitle(f"Level-wise RMSE vs lead time — {label}", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{out}/rmse_levelwise_heatmap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def p_mean(mean_e, mean_t, lon, lat, wet, label, out):
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for row, k in enumerate(("SST", "SSS", "SSH")):
        te = np.where(wet > 0, mean_e[k], np.nan)
        tt = np.where(wet > 0, mean_t[k], np.nan)
        vmin, vmax = np.nanpercentile(tt, [2, 98])
        lim = np.nanpercentile(np.abs(te - tt), 98)
        for col, (field, ttl, kw) in enumerate(
            [
                (tt, "truth", dict(vmin=vmin, vmax=vmax, cmap="viridis")),
                (te, "emul", dict(vmin=vmin, vmax=vmax, cmap="viridis")),
                (te - tt, "emul−truth", dict(vmin=-lim, vmax=lim, cmap="RdBu_r")),
            ]
        ):
            ax = axes[row, col]
            im = ax.pcolormesh(lon, lat, field, shading="auto", **kw)
            ax.set_title(f"{k} {ttl}")
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"Time-mean surface fields — {label}", y=1.005)
    fig.tight_layout()
    fig.savefig(f"{out}/mean_surface_fields.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def p_snapshots(snaps, lon, lat, wet, label, out):
    hs = sorted(snaps)
    fig, axes = plt.subplots(len(hs), 3, figsize=(15, 4.0 * len(hs)))
    if len(hs) == 1:
        axes = axes[None, :]
    for r, h in enumerate(hs):
        te = np.where(wet > 0, snaps[h][0], np.nan)
        tt = np.where(wet > 0, snaps[h][1], np.nan)
        vmin, vmax = np.nanpercentile(tt, [2, 98])
        lim = np.nanpercentile(np.abs(te - tt), 98)
        for c, (field, ttl, kw) in enumerate(
            [
                (tt, "truth", dict(vmin=vmin, vmax=vmax, cmap="viridis")),
                (te, "emul", dict(vmin=vmin, vmax=vmax, cmap="viridis")),
                (te - tt, "diff", dict(vmin=-lim, vmax=lim, cmap="RdBu_r")),
            ]
        ):
            ax = axes[r, c]
            im = ax.pcolormesh(lon, lat, field, shading="auto", **kw)
            ax.set_title(f"SST {ttl}  day {h}")
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"SST snapshots at lead times — {label}", y=1.002)
    fig.tight_layout()
    fig.savefig(f"{out}/sst_snapshots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def p_spectra(spec, lon, lat, label, out):
    latc = lat[lat.shape[0] // 2, :] if lat.ndim == 2 else lat
    lonc = lon[:, lon.shape[1] // 2] if lon.ndim == 2 else lon
    dx, dy = estimate_grid_spacing_km(latc, lonc)

    def ke(u, v):
        mu = np.isfinite(u)
        mv = np.isfinite(v)
        up = np.where(mu, u, 0.0).astype(np.float64)
        up[mu] -= np.nanmean(u[mu])
        vp = np.where(mv, v, 0.0).astype(np.float64)
        vp[mv] -= np.nanmean(v[mv])
        uf, kx, ky, _ = compute_2d_fft_normalized(up, dx, dy, True)
        vf, _, _, _ = compute_2d_fft_normalized(vp, dx, dy, True)
        return azimuthal_average(0.5 * (np.abs(uf) ** 2 + np.abs(vf) ** 2), kx, ky)

    hs = sorted(spec)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ct = plt.cm.Blues(np.linspace(0.4, 0.9, len(hs)))
    ce = plt.cm.Reds(np.linspace(0.4, 0.9, len(hs)))
    kref = mid = good = None
    for i, h in enumerate(hs):
        ue, ve, ut, vt = spec[h]
        k, kt = ke(ut, vt)
        _, keu = ke(ue, ve)
        good = np.isfinite(kt) & (k > 0)
        ax.loglog(k[good], kt[good], color=ct[i], lw=1.2, label=f"truth d{h}")
        ax.loglog(k[good], keu[good], color=ce[i], lw=1.2, ls="--", label=f"emul d{h}")
        kref, mid = k[good], np.nanmedian(kt[good])
    if kref is not None:
        km = np.median(kref)
        ax.loglog(kref, mid * (kref / km) ** (-3), "k:", lw=0.7, alpha=0.6, label=r"$k^{-3}$")
    ax.set_xlabel("wavenumber [cycles/km]")
    ax.set_title(f"Surface KE spectrum — {label}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out}/ke_spectra_uv.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Open once to learn the step count, then reopen cropped to the truth grid — the OBC halo
    # is only detectable relative to the truth's own (yh, xh) sizes.
    n_steps = load_emulator(args.rollout_dir).sizes["time"]
    truth = load_truth(n_steps)
    emul = load_emulator(args.rollout_dir, truth_shape=(truth.sizes["yh"], truth.sizes["xh"]))
    levels = np.asarray(truth["z1_l"].values, float)
    nlev = len(levels)
    print(f"[{args.label}] steps={n_steps} levels={nlev}", flush=True)

    g = gather(emul, truth, n_steps, nlev)
    np.savez(f"{args.out}/rmse_curves.npz", days=np.arange(1, n_steps + 1), **g["surf"])
    np.savez(f"{args.out}/rmse_levelwise.npz", levels=levels, **{f"lw_{v}": g["lw"][v] for v in VARS3D})

    lon, lat, wet = _geo()
    p_rmse_curves(g["surf"], n_steps, args.label, args.out)
    p_depth(g["lw"], levels, args.label, args.out)
    p_heatmap(g["lw"], levels, n_steps, args.label, args.out)
    p_mean(g["mean_e"], g["mean_t"], lon, lat, wet, args.label, args.out)
    p_snapshots(g["snaps"], lon, lat, wet, args.label, args.out)
    p_spectra(g["spec"], lon, lat, args.label, args.out)
    print(f"[{args.label}] wrote plots to {args.out}", flush=True)


if __name__ == "__main__":
    main()
