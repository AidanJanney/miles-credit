"""
eval_rmom6_compare.py
=====================
Overlay the 4 multistep rMOM6 experiments (levelwise/pointwise x tendency/notendency)
on shared axes, using the per-model npz files written by eval_rmom6_rollout.py.

  - Surface RMSE (SST/SSS/SSH) vs lead time, all 4 models overlaid
  - Time-averaged RMSE vs depth (thetao/so/uo/vo), all 4 models overlaid

Usage:
  python scripts/eval_rmom6_compare.py --eval-root /glade/.../eval_rmom6 --out /glade/.../eval_rmom6/_compare
"""

from __future__ import annotations
import argparse
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = [
    "levelwise_multistep_notendency",
    "levelwise_multistep_tendency",
    "pointwise_multistep_notendency",
    "pointwise_multistep_tendency",
]
COLORS = {m: c for m, c in zip(MODELS, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])}
VARS3D = ["thetao", "so", "uo", "vo"]
HORIZONS = [30, 60, 120, 365, 729]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    curves = {}
    lw = {}
    for m in MODELS:
        cp = os.path.join(args.eval_root, m, "rmse_curves.npz")
        lp = os.path.join(args.eval_root, m, "rmse_levelwise.npz")
        if os.path.exists(cp):
            curves[m] = np.load(cp)
        if os.path.exists(lp):
            lw[m] = np.load(lp)
    if not curves:
        raise SystemExit(f"No rmse_curves.npz found under {args.eval_root}/<model>/")

    # ---- Surface RMSE overlay ----
    units = {"SST": "°C", "SSS": "PSU", "SSH": "m"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, k in zip(axes, ("SST", "SSS", "SSH")):
        for m, d in curves.items():
            ax.plot(d["days"], d[k], lw=1.3, color=COLORS[m], label=m.replace("_multistep", ""))
        for h in HORIZONS:
            ax.axvline(h, color="gray", ls=":", lw=0.5, alpha=0.4)
        ax.set_xlabel("lead time [days]")
        ax.set_ylabel(f"RMSE [{units[k]}]")
        ax.set_title(k)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Surface RMSE vs lead time — 4 multistep experiments", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{args.out}/compare_rmse_surface.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # log-y variant (tendency loss scales differ hugely)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, k in zip(axes, ("SST", "SSS", "SSH")):
        for m, d in curves.items():
            ax.semilogy(d["days"], d[k], lw=1.3, color=COLORS[m], label=m.replace("_multistep", ""))
        ax.set_xlabel("lead time [days]")
        ax.set_ylabel(f"RMSE [{units[k]}] (log)")
        ax.set_title(k)
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7)
    fig.suptitle("Surface RMSE vs lead time (log-y) — 4 multistep experiments", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{args.out}/compare_rmse_surface_logy.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---- Depth RMSE overlay ----
    if lw:
        fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
        for ax, v in zip(axes, VARS3D):
            for m, d in lw.items():
                prof = np.nanmean(d[f"lw_{v}"], axis=1)
                ax.plot(
                    prof, d["levels"], lw=1.4, color=COLORS[m], marker="o", ms=2.5, label=m.replace("_multistep", "")
                )
            ax.invert_yaxis()
            ax.set_xlabel(f"RMSE {v}")
            ax.grid(alpha=0.3)
            ax.set_title(v)
        axes[0].set_ylabel("depth [m]")
        axes[0].legend(fontsize=7)
        fig.suptitle("Time-averaged RMSE vs depth — 4 multistep experiments", y=1.02)
        fig.tight_layout()
        fig.savefig(f"{args.out}/compare_rmse_depth.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    # ---- RMSE-at-horizon table (SST/SSS/SSH) ----
    lines = ["model," + ",".join(f"SST_d{h},SSS_d{h},SSH_d{h}" for h in HORIZONS)]
    for m, d in curves.items():
        n = len(d["days"])
        row = [m]
        for h in HORIZONS:
            if h <= n:
                row += [f"{d['SST'][h - 1]:.3f}", f"{d['SSS'][h - 1]:.3f}", f"{d['SSH'][h - 1]:.3f}"]
            else:
                row += ["", "", ""]
        lines.append(",".join(row))
    with open(f"{args.out}/rmse_at_horizons.csv", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote comparison to", args.out)


if __name__ == "__main__":
    main()
