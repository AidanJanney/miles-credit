"""
benchmark_rmom6.py
------------------
Phase-level timing of a gen2 regional-ocean training iteration: where does the wall clock
actually go?

Motivation
==========
The 8-experiment sweep was running at ~500 s/iter on 8 GPUs, and the configs carried a
``# compute-bound at this batch size`` note. That note is a hypothesis, not a measurement.
This script measures it, by decomposing one training iteration into the same phases
``TrainerERA5Gen2.train_one_epoch`` runs, and separately isolating the two extremes:

    STAGE 1  dataloader only   -- iterate batches, no model at all
    STAGE 2  model only        -- one real batch replayed, no I/O at all (fwd+bwd+step)
    STAGE 3  full iteration    -- the real loop, per-phase timers:
                                    data      next(dl), i.e. waiting on the dataloader
                                    preblocks normalize -> fill -> concat
                                    forward   model forward
                                    postblocks reconstruct -> denorm -> clamp -> mask -> obc
                                    backward  loss + backward
                                    optimizer grad clip + step

If STAGE 1 ~= STAGE 3, training is I/O-bound and no amount of GPU is going to help. If
STAGE 2 ~= STAGE 3, it is genuinely compute-bound. The per-phase table then says which
specific block to attack.

Every GPU phase is wrapped in ``torch.cuda.synchronize()``, without which CUDA's async
dispatch would misattribute forward/backward time to whatever phase happens to sync next
(usually ``.item()`` on the loss) and make the model look free.

Usage
=====
Needs a GPU -- run it on a GPU node, not a login node::

    # interactive
    python -u scripts/benchmark_rmom6.py -c config/rmom6_bench_multistep.yml --iters 3

    # batch (see scripts/derecho_benchmark_rmom6.sh)
    qsub scripts/derecho_benchmark_rmom6.sh

Single-rank by design: DDP all-reduce is a separate axis, and phase attribution is clearest
without it. Compare the singlestep and multistep bench configs to see how cost scales with
``forecast_len``.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from contextlib import contextmanager

import torch
import yaml

from credit.models import load_model
from credit.losses import load_loss
from credit.preblock import apply_preblocks
from credit.postblock import apply_postblocks
from credit.trainers.rollout_utils import assemble_rollout_batch
from credit.trainers.trainer_gen2 import TrainerERA5Gen2

# The gen2 loaders live in trainers.utils; credit.datasets.load_dataset_and_dataloader is the
# gen1 pair and sys.exit()s on a gen2 config (it expects credit_main_parser to have run).
from credit.trainers.utils import inject_flat_var_keys, load_dataset, load_dataloader


class Timer:
    """Accumulates per-phase wall time, synchronizing CUDA so GPU work lands in its own phase."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = defaultdict(float)

    @contextmanager
    def __call__(self, phase: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.totals[phase] += time.perf_counter() - t0

    def report(self, title: str, n_iters: int) -> None:
        total = sum(self.totals.values())
        print(f"\n  {title}  ({total / n_iters:.1f} s/iter over {n_iters} iters)")
        print(f"    {'phase':<12} {'s/iter':>9} {'% of iter':>10}")
        print(f"    {'-' * 12} {'-' * 9} {'-' * 10}")
        for phase, secs in sorted(self.totals.items(), key=lambda kv: -kv[1]):
            print(f"    {phase:<12} {secs / n_iters:>9.2f} {100 * secs / total:>9.1f}%")


def cycle(loader):
    while True:
        yield from loader


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--iters", type=int, default=3, help="Timed iterations (after 1 untimed warmup).")
    ap.add_argument("--skip-model-only", action="store_true", help="Skip STAGE 2 (saves a warmup).")
    args = ap.parse_args()

    # train_gen2.py deliberately bypasses credit_main_parser (gen2 configs are read as-is);
    # mirror that exactly so the benchmark exercises the same code path training does.
    with open(args.config) as f:
        conf = yaml.safe_load(f)
    inject_flat_var_keys(conf)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("!! No GPU visible -- forward/backward numbers will be meaningless. Run on a GPU node.")

    bs = conf["trainer"]["train_batch_size"]
    fl = conf["data"]["forecast_len"]
    workers = conf["trainer"]["thread_workers"]
    print(f"config          : {args.config}")
    print(f"batch_size      : {bs}")
    print(f"forecast_len    : {fl}   (-> {fl} dataloader batches consumed per iteration)")
    print(f"thread_workers  : {workers}")

    dataset = load_dataset(conf, is_train=True)
    loader = load_dataloader(conf, dataset, rank=0, world_size=1, is_train=True)

    model = load_model(conf).to(device)

    # Production wraps the model with distributed_model_wrapper_gen2, which (among other
    # things) applies activation checkpointing when trainer.activation_checkpoint is set.
    # Skipping it here would OOM at batch_size 16 AND overstate the forward/backward cost, so
    # apply the same AC directly -- no process group needed, unlike the full wrapper.
    if conf["trainer"].get("activation_checkpoint", False):
        from credit.parallel.fsdp2 import _apply_activation_checkpointing

        _apply_activation_checkpointing(model)
        print("activation_ckpt : True (matching production)")

    trainer = TrainerERA5Gen2(model, 0, conf)
    criterion = load_loss(conf)
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf["trainer"]["learning_rate"])

    # ---------------------------------------------------------------- STAGE 1: dataloader only
    dl = cycle(loader)
    next(dl)  # warmup: spin up workers, page in metadata
    t0 = time.perf_counter()
    n_batches = args.iters * fl  # one full iteration consumes forecast_len batches
    for _ in range(n_batches):
        next(dl)
    data_only = (time.perf_counter() - t0) / args.iters
    print(f"\nSTAGE 1  dataloader only : {data_only:7.1f} s/iter  ({n_batches} batches, {workers} workers)")

    # ---------------------------------------------------------------- STAGE 2: model only
    model_only = float("nan")
    if not args.skip_model_only:
        # Fresh iterator, not STAGE 1's `dl`: DistributedMultiStepBatchSampler yields a full
        # IC sample only on every fl-th pull (the rest are forcing-only rollout-continuation
        # samples, far fewer channels -- see MultiStepBatchSamplerSubset in credit/samplers.py).
        # STAGE 1's one warmup pull leaves `dl` out of phase with fl for fl > 1, so reusing it
        # here would grab a continuation sample and crash the model on channel count. A brand
        # new iterator always starts at the sampler's i=0, i.e. a full IC sample.
        batch = next(cycle(loader))
        prepped = apply_preblocks(trainer.step_preblocks, apply_preblocks(trainer.ic_preblocks, batch, device=device), device=device)
        x = prepped["x"].detach()
        y = prepped["y"].detach()
        print(f"         (model input x: {tuple(x.shape)}, target y: {tuple(y.shape)})")

        for _ in range(2):  # warmup: cudnn autotune + allocator
            loss = criterion(y, model(x)).mean()
            loss.backward()
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize() if device.type == "cuda" else None

        t0 = time.perf_counter()
        for _ in range(args.iters):
            # forecast_len forward/backward passes = one iteration's worth of compute
            for _ in range(fl):
                loss = criterion(y, model(x)).mean()
                loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        model_only = (time.perf_counter() - t0) / args.iters
        peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
        print(f"STAGE 2  model only      : {model_only:7.1f} s/iter  ({fl} fwd+bwd, no I/O, peak {peak:.1f} GB)")

        # Don't let STAGE 2's cached batch inflate STAGE 3's memory footprint.
        del x, y, prepped, batch, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    # ---------------------------------------------------------------- STAGE 3: full instrumented loop
    # Fresh iterator for the same reason as STAGE 2: t==1 in the loop below must land on a full
    # IC sample (sampler i==0), and STAGE 1/2's iterators are left out of phase for fl > 1.
    dl = cycle(loader)
    timer = Timer()
    model.train()
    for it in range(args.iters + 1):  # first iteration is an untimed warmup
        if it == 1:
            timer.totals.clear()
        full: dict = {}
        optimizer.zero_grad(set_to_none=True)

        for t in range(1, fl + 1):
            with timer("data"):
                batch = next(dl)

            with timer("preblocks"):
                if t == 1:
                    full["ic_preprocessed"] = apply_preblocks(trainer.ic_preblocks, batch, device=device)
                    full.update(apply_preblocks(trainer.step_preblocks, full["ic_preprocessed"], device=device))
                else:
                    full.update(
                        apply_preblocks(trainer.step_preblocks, assemble_rollout_batch(full, batch), device=device)
                    )

            with timer("forward"):
                full["y_pred"] = model(full["x"])

            with timer("postblocks"):
                full = apply_postblocks(trainer.step_postblocks, full)

            if t in trainer.backprop_on_timestep:
                with timer("backward"):
                    loss = criterion(full["y"].float().to(full["y_pred"].dtype), full["y_pred"]).mean()
                    loss.backward()

        with timer("optimizer"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    timer.report("STAGE 3  full iteration", args.iters)

    # ---------------------------------------------------------------- verdict
    # Judge on STAGE 3's `data` share, NOT on STAGE 1. STAGE 1 consumes only a few batches, and
    # `thread_workers x prefetch_factor` of them are already sitting in the prefetch queue when
    # it starts -- so it clocks the queue draining, not the rate workers can actually sustain,
    # and badly understates I/O. STAGE 3 runs long enough to exhaust that queue and block on the
    # workers, which is what training does.
    full_iter = sum(timer.totals.values()) / args.iters
    data_share = timer.totals["data"] / args.iters / full_iter
    compute_share = (timer.totals["forward"] + timer.totals["backward"]) / args.iters / full_iter
    print(f"\n  STAGE 3 data share    : {100 * data_share:.0f}%  (waiting on the dataloader)")
    print(f"  STAGE 3 compute share : {100 * compute_share:.0f}%  (forward + backward)")
    print(f"  (STAGE 1's {data_only:.1f} s/iter is a prefetch-queue drain, not a throughput number -- ignore it)")
    print(f"  => {'I/O-BOUND' if data_share > 0.5 else 'COMPUTE-BOUND'}")


if __name__ == "__main__":
    main()
