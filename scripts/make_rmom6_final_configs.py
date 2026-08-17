#!/usr/bin/env python
"""Generate the 8 `_final` rMOM6 experiment configs from the `_full` ones.

The `_full` grid was a scale/plumbing test: batches_per_epoch 4 (multistep) and 10
(singlestep) against a real epoch of 52 iterations, i.e. 8% and 20% of the data per epoch,
and validation on 2 samples. These configs keep the experiment design identical and change
only the training budget, the LR schedule, the PBS chaining, and the inference block.

Derived from `_full` rather than hand-written so the parts that define the experiment --
preblocks (including the 11-key input-normalization fix), postblocks, model geometry, scaler
paths, loss -- cannot drift. The script hard-fails if the normalization fix is missing.

What changes, and why:

batches_per_epoch: 0
    0 means "use the whole dataset" (trainer_gen2.py:261). One epoch is 52 iterations:
    6572 samples / 8 ranks = 822 per rank, / train_batch_size 16 = 52.

valid_batch_size 1 -> 16, valid_batches_per_epoch 2 -> 0
    Validation was 2 samples, which is why valid_acc was noise. At batch size 16 across 8
    ranks the full 718-sample validation set is only 6 iterations, so "all of it" is also
    the cheap option (~45 s/epoch).

epochs: 50 everywhere
    `_full` had pointwise_multistep_tendency at 20 while its three multistep siblings were
    at 50, which would have made that one arm non-comparable.

scheduler: cosine-annealing -> linear-warmup-cosine
    Plain CosineAnnealingLR takes no warmup argument. linear-warmup-cosine is stepped
    per-batch (credit/scheduler.py:8 update_on_batch, honored at trainer_gen2.py:441), so
    total_steps is in ITERATIONS: 50 epochs * 52 = 2600. The warmup matters most for the
    tendency arm, which starts at loss ~5e9 and otherwise takes an enormous first step.

loss.mask_missing_targets: True
    gen1 computed criterion(y * wet_mask, y_pred) (trainer_om4_samudra.py:275). gen2 had no
    equivalent, so about half of every 3D field -- land and below-bathymetry cells, which
    fill_values sets to a constant 0 -- was inside the loss. The trainer now means the loss over
    cells the raw target actually defines (trainer_gen2.py, _reduce_loss). Loss magnitudes
    roughly double against the `_full` logs; metrics are unaffected.

amp stays False
    trainer_gen2 wraps the forward in torch.autocast, which defaults to fp16. GradScaler *is*
    wired in (credit/trainers/utils.py:647) and the reduction promotes to fp32, but the tendency
    runs (losses 1e4-1e9) can saturate fp16's 65504 in the forward pass itself. Not worth the
    throughput until trainer.amp_dtype: bfloat16 exists.

num_epoch / walltime
    12 h jobs (Derecho main maximum) instead of 4 h: 12 epochs/job multistep (~10.9 h at
    54.6 min/epoch), 25/job singlestep (~7.6 h at 18.2 min/epoch). That is 5 and 2 chain
    links instead of 25 and 2 -- every afterok boundary is another chance at the PBS -32
    abort that ended the `_full` multistep runs.

Usage::

    python scripts/make_rmom6_final_configs.py
    python scripts/make_rmom6_final_configs.py --epochs 100     # rescales total_steps too
"""

from __future__ import annotations

import argparse
import glob
import os

import yaml

REPO = "/glade/work/ajanney/RegionalEmulation_v2/miles-credit-regional"
FULL_GLOB = os.path.join(REPO, "config", "rmom6_regional_*_full.yml")
OUT_DIR = os.path.join(REPO, "config")

ITERS_PER_EPOCH = 52  # 6572 train samples / 8 ranks / train_batch_size 16
MIN_PER_EPOCH = {1: 18.2, 3: 54.6}  # forecast_len -> measured wall minutes for a full epoch

PREP = "/glade/derecho/scratch/ajanney/rmom6_regional/preprocessed"
# Only north and east carry data: obc_south / obc_west are entirely land on this domain
# (their thetao bottoms out at exactly 0.0, the land fill, while north/east are 1.65/1.43).
# yh increases northward and xh eastward, so north is the last row and east the last column.
OBC_EDGES = {"rMOM6_obc_north": ("north", "lat_end"), "rMOM6_obc_east": ("east", "lon_end")}
OBC_PAD_LAT = [0, 1]  # rows appended to 457 -> 458
OBC_PAD_LON = [0, 1]  # cols appended to 759 -> 760
MODEL_PAD_LAT = [27, 27]  # 458 + 54 = 512
MODEL_PAD_LON = [4, 4]  # 760 +  8 = 768


def _wire_obc_halo(conf: dict) -> None:
    """Make the open boundary an *input* (gen1's approach) instead of a post-hoc overwrite.

    Each ring becomes its own gen2 source, so the per-source ``normalize`` preblock standardizes
    it with its own statistics (boundary cells are not distributed like the interior); then
    ``ocean_obc_halo`` — which runs after ``normalize`` and before ``concat`` — writes the
    normalized rings into a one-cell halo. Declaring them under ``dynamic_forcing`` is what makes
    ``assemble_rollout_batch`` re-read them from the dataset every rollout step, rather than
    carrying the model's own halo prediction forward: the halo is prescribed, not predicted.

    ``ocean_obc_nudge`` is removed — overwriting the outer ring after the forward pass is exactly
    what this replaces, and leaving both would stomp the halo with a second copy.
    """
    src = conf["data"]["source"]
    interior = src["rMOM6"]
    for source, (edge, _) in OBC_EDGES.items():
        src[source] = {
            "dataset_type": "rmom6",
            "level_coord": interior["level_coord"],
            "levels": interior["levels"],
            "variables": {
                # dynamic_forcing => refreshed from disk each rollout step (see docstring).
                "dynamic_forcing": {
                    "vars_3D": ["uo", "vo", "thetao", "so"],
                    "vars_2D": ["SSH"],
                    "path": f"{PREP}/obc_{edge}.zarr",
                }
            },
        }
    # validation_data carries its own source dict; keep the two in sync.
    if isinstance(conf.get("validation_data"), dict) and "source" in conf["validation_data"]:
        for source in OBC_EDGES:
            conf["validation_data"]["source"][source] = src[source]

    # normalize must cover the new keys or pointwise silently skips them (leaving raw values in
    # the input) and bridgescaler asserts. scripts/build_rmom6_obc_stats.py writes these.
    norm_vars = conf["preblocks"]["per_step"]["normalize"]["args"]["variables"]
    for source in OBC_EDGES:
        for v in ["uo", "vo", "thetao", "so"]:
            norm_vars.append(f"{source}/dynamic_forcing/3d/{v}")
        norm_vars.append(f"{source}/dynamic_forcing/2d/SSH")

    # Rebuild per_step preblocks so obc_halo lands between fill_land and concat.
    per_step = conf["preblocks"]["per_step"]
    rebuilt = {}
    for name, block in per_step.items():
        if name == "concat":
            rebuilt["obc_halo"] = {
                "type": "ocean_obc_halo",
                "args": {
                    "interior_source": "rMOM6",
                    "pad_lat": OBC_PAD_LAT,
                    "pad_lon": OBC_PAD_LON,
                    "corner": "lon",
                    # Pad the target too, so y and y_pred share the halo grid; the halo is then
                    # dropped from the loss by loss.exclude_border below.
                    "data_types": ["input", "target"],
                    "edges": {s: {"source": s, "edge": e} for s, (_, e) in OBC_EDGES.items()},
                },
            }
        rebuilt[name] = block
    conf["preblocks"]["per_step"] = rebuilt

    # The halo is prescribed, so scoring the model on reproducing it is meaningless.
    conf["loss"]["exclude_border"] = {"lat": OBC_PAD_LAT, "lon": OBC_PAD_LON}

    post = conf["postblocks"]["per_step"]
    post.pop("obc_nudge", None)
    post["wet_mask"]["args"]["halo_pad"] = [OBC_PAD_LAT, OBC_PAD_LON]

    m = conf["model"]
    m["image_height"] += OBC_PAD_LAT[0] + OBC_PAD_LAT[1]
    m["image_width"] += OBC_PAD_LON[0] + OBC_PAD_LON[1]
    m["padding_conf"]["pad_lat"] = MODEL_PAD_LAT
    m["padding_conf"]["pad_lon"] = MODEL_PAD_LON
    padded_h = m["image_height"] + sum(MODEL_PAD_LAT)
    padded_w = m["image_width"] + sum(MODEL_PAD_LON)
    # interp: False means the network never resizes, so a padded grid that does not divide
    # cleanly through four stride-2 stages surfaces as a shape error deep in the forward.
    for stage, gwin in enumerate(m["global_window_size"], start=1):
        h, w = padded_h >> stage, padded_w >> stage
        for label, win in (("global", gwin), ("local", m["local_window_size"])):
            if h % win or w % win:
                raise SystemExit(
                    f"padded grid {padded_h}x{padded_w}: stage {stage} is {h}x{w}, "
                    f"not divisible by {label}_window_size {win}."
                )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--warmup-steps", type=int, default=100, help="Iterations of linear LR warmup.")
    ap.add_argument("--job-hours", type=float, default=12.0, help="Walltime per chained PBS job.")
    ap.add_argument(
        "--no-obc", action="store_true", help="Skip the OBC halo wiring (keep the 457x759 grid + obc_nudge postblock)."
    )
    ap.add_argument("--rollout-days", type=int, default=729, help="Final rollout length (2018-01-01 -> 2019-12-30).")
    args = ap.parse_args()

    configs = sorted(glob.glob(FULL_GLOB))
    if len(configs) != 8:
        raise SystemExit(f"expected 8 _full configs, found {len(configs)}")

    total_steps = args.epochs * ITERS_PER_EPOCH
    print(f"epochs={args.epochs}  iters/epoch={ITERS_PER_EPOCH}  total_steps={total_steps}\n")

    for path in configs:
        conf = yaml.safe_load(open(path))
        tag = os.path.basename(path).replace("rmom6_regional_", "").replace("_full.yml", "")

        n_norm = len(conf["preblocks"]["per_step"]["normalize"]["args"]["variables"])
        if n_norm != 11:
            raise SystemExit(f"{tag}: preblocks.normalize lists {n_norm} vars, expected 11")

        conf["save_loc"] = conf["save_loc"].replace("_full", "_final")
        fl = conf["data"]["forecast_len"]

        # How many epochs fit in one job, leaving ~10% headroom for startup and checkpointing.
        epochs_per_job = max(1, int(args.job_hours * 60 * 0.9 / MIN_PER_EPOCH[fl]))
        epochs_per_job = min(epochs_per_job, args.epochs)
        n_links = -(-args.epochs // epochs_per_job)

        t = conf["trainer"]
        t["batches_per_epoch"] = 0  # full epoch
        t["valid_batch_size"] = 16
        t["valid_batches_per_epoch"] = 0  # full validation set = 6 iterations
        t["epochs"] = args.epochs
        t["num_epoch"] = epochs_per_job
        t["start_epoch"] = 0
        t["amp"] = False
        t["scheduler"] = {
            "scheduler_type": "linear-warmup-cosine",
            "warmup_steps": args.warmup_steps,
            "total_steps": total_steps,
            "min_lr": 1.0e-6,
        }

        # Land / below-bathymetry cells are ~50% of every 3D field; keep them out of the loss.
        conf["loss"]["mask_missing_targets"] = True

        if not args.no_obc:
            _wire_obc_halo(conf)

        conf["pbs"] = {
            **conf.get("pbs", {}),
            "job_name": f"rmom6_{tag}_final",
            "walltime": f"{int(args.job_hours):02d}:00:00",
            "queue": "main",
        }

        conf["inference"] = {
            "run_mode": "single",
            "mode": "none",
            "save_forecast": f"{conf['save_loc']}/rollout_2018_{args.rollout_days}d",
            "single_forecast": {
                "forecast_length": f"{args.rollout_days}d",
                # Forcing and truth both end 2019-12-31, so 729 days from here is the
                # longest fully-verifiable out-of-sample rollout available.
                "start_datetime": "2018-01-01 12:00:00",
            },
            "output": {
                "format": "netcdf",
                "output_interval": None,
                "group_by": "monthly",
                "variables": None,
                "metadata": None,
                "encoding": {"dtype": "float32", "zlib": True, "complevel": 4},
            },
        }

        out = os.path.join(OUT_DIR, f"rmom6_regional_{tag}_final.yml")
        with open(out, "w") as f:
            f.write(f"# GENERATED by scripts/make_rmom6_final_configs.py from {os.path.basename(path)}\n")
            f.write(f"# Full-epoch training: {args.epochs} epochs x {ITERS_PER_EPOCH} iters = {total_steps} steps.\n")
            f.write("# Do not hand-edit; change the generator and regenerate so all 8 stay in sync.\n")
            yaml.dump(conf, f, sort_keys=False, default_flow_style=False, width=120)

        hrs = args.epochs * MIN_PER_EPOCH[fl] / 60
        print(
            f"{tag:38s} fcst={fl}  {epochs_per_job:2d} ep/job x {n_links} links  "
            f"~{hrs:5.1f} h total  -> {os.path.basename(out)}"
        )


if __name__ == "__main__":
    main()
