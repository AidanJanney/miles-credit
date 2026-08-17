#!/usr/bin/env python
"""Report validation skill at EACH rollout step for a trained rMOM6 checkpoint.

Motivation: trainer_gen2 reports one ACC per sample. Before the averaging change it was
the final step alone; after it, the mean over steps. Neither tells you whether skill is
uniform across the rollout or collapses after step 1 -- which is exactly the open question
for the multistep runs, whose valid_acc sat at ~1e-4 while singlestep reached 0.91.

This runs the same validation rollout the trainer runs (identical preblocks, postblocks and
assemble_rollout_batch feedback) on a single GPU, and prints ACC / MSE / MAE per step:

  * ACC ~0.9 at step 1 decaying toward 0 by step 3  -> the feedback loop diverges
  * ACC ~0 already at step 1                        -> the model never learned the map,
                                                       so multistep TRAINING is at fault

A persistence baseline is printed alongside. Over 3 days the ocean barely changes, so
persistence should score ACC ~0.99; if the model is below it, the model is worse than
doing nothing, which error accumulation alone cannot produce. Persistence assumes the
prognostic channels lead the input tensor (FIELD_TYPE_RANK in channel_layout.py orders
prognostic < static < dynamic_forcing < diagnostic), so x[:, :y.shape[1]] is the current
state. That assumption is self-checking: if it is wrong, step-1 persistence ACC will not
come out near 1. It is meaningless for the `_tendency` configs, where the target is not the
state, so it is reported only when --no-persistence is absent and should be ignored there.

Usage::

    python scripts/diagnose_rmom6_per_step.py \\
        -c config/rmom6_regional_pointwise_multistep_notendency_full.yml --samples 4

Runs on one GPU; submit with --launch to send it to the develop queue.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from credit.datasets.multi_source import MultiSourceDataset
from credit.metrics import LatWeightedMetrics
from credit.postblock import apply_postblocks, build_postblocks
from credit.preblock import apply_preblocks, build_preblocks
from credit.samplers import MultiStepBatchSamplerSubset
from credit.trainers.rollout_utils import assemble_rollout_batch, load_model_for_inference
from credit.trainers.utils import inject_flat_var_keys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--samples", type=int, default=4, help="Validation samples to average over.")
    ap.add_argument("--steps", type=int, default=None, help="Rollout steps (default: config valid_forecast_len).")
    ap.add_argument("--no-persistence", action="store_true", help="Skip the persistence baseline.")
    args = ap.parse_args()

    conf = yaml.safe_load(open(args.config))
    conf["save_loc"] = os.path.expandvars(conf["save_loc"])
    ckpt = os.path.join(conf["save_loc"], "checkpoint.pt")
    if not os.path.isfile(ckpt):
        sys.exit(f"no checkpoint at {ckpt}")

    # load_model_for_inference reads conf["inference"]["mode"]; force single-process.
    conf.setdefault("inference", {})["mode"] = "none"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_conf = dict(conf.get("validation_data") or conf["data"])
    data_conf.setdefault("source", conf["data"]["source"])
    n_steps = args.steps or data_conf.get("valid_forecast_len") or data_conf["forecast_len"]
    data_conf["forecast_len"] = n_steps

    print(f"config     : {os.path.basename(args.config)}")
    print(f"checkpoint : {ckpt}")
    print(f"steps      : {n_steps}   samples: {args.samples}   device: {device}\n", flush=True)

    dataset = MultiSourceDataset(data_conf, return_target=True, label="valid")
    sampler = MultiStepBatchSamplerSubset(
        dataset=dataset,
        batch_size=1,
        index_subset=list(range(args.samples)),
        num_forecast_steps=n_steps,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)

    ic_pre = build_preblocks(conf.get("preblocks", {}), phase="ic_only")
    step_pre = build_preblocks(conf.get("preblocks", {}), phase="per_step")
    step_post = build_postblocks(conf.get("postblocks", {}), phase="per_step")

    model = load_model_for_inference(conf, device)
    model.eval()
    # LatWeightedMetrics reads the flat Gen1-style variable lists, which train_gen2 injects
    # from the nested Gen2 source config before constructing it.
    inject_flat_var_keys(conf)
    metrics = LatWeightedMetrics(conf)

    # acc[step] accumulates across samples; same for the rest.
    acc, mse, mae, pacc = ([[] for _ in range(n_steps)] for _ in range(4))

    it = iter(loader)
    with torch.no_grad():
        for s in range(args.samples):
            fdd: dict = {}
            for t in range(1, n_steps + 1):
                batch = next(it)
                if t == 1:
                    fdd["ic_raw"] = batch["input"]
                    fdd["ic_preprocessed"] = apply_preblocks(ic_pre, batch, device=device)
                    fdd.update(apply_preblocks(step_pre, fdd["ic_preprocessed"], device=device))
                else:
                    fdd.update(apply_preblocks(step_pre, assemble_rollout_batch(fdd, batch), device=device))

                x = fdd["x"]
                fdd["y_pred"] = model(x)
                if fdd["y_pred"].dim() == 5:
                    fdd["y_pred"] = fdd["y_pred"].flatten(1, 2)
                fdd = apply_postblocks(step_post, fdd)

                y = fdd["y"].float()
                m = metrics(fdd["y_pred"].float(), y)
                acc[t - 1].append(m["acc"])
                mse[t - 1].append(m["mse"])
                mae[t - 1].append(m["mae"])

                if not args.no_persistence:
                    nc = y.shape[1]
                    xf = x.flatten(1, 2) if x.dim() == 5 else x
                    pacc[t - 1].append(metrics(xf[:, :nc].float(), y)["acc"])
            print(f"  sample {s + 1}/{args.samples} done", flush=True)

    print(f"\n{'step':>5} {'ACC':>9} {'MSE':>10} {'MAE':>9}" + ("" if args.no_persistence else f" {'ACC_persist':>12}"))
    print("-" * (36 if args.no_persistence else 49))
    for t in range(n_steps):
        row = f"{t + 1:5d} {np.mean(acc[t]):9.4f} {np.mean(mse[t]):10.4f} {np.mean(mae[t]):9.4f}"
        if not args.no_persistence:
            row += f" {np.mean(pacc[t]):12.4f}"
        print(row)
    print(
        f"\nmean over steps: ACC={np.mean([np.mean(a) for a in acc]):.4f}  "
        f"MSE={np.mean([np.mean(m) for m in mse]):.4f}   "
        f"(final step alone: ACC={np.mean(acc[-1]):.4f})"
    )
    print("\nMSE reference: predicting a constant zero in normalized space scores exactly 1.0.")


if __name__ == "__main__":
    main()
