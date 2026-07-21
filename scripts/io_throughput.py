"""Sustained dataloader throughput at thread_workers 0 vs 4, at production batch size.

Why this exists
===============
``scripts/benchmark_rmom6.py``'s STAGE 3 runs ``--iters 3``. For ``forecast_len=1`` that
consumes 3 batches, while ``thread_workers=4 x prefetch_factor=2`` leaves 8 already queued
before timing starts -- so it never blocks on I/O once, and its "s/iter" is a prefetch-queue
drain, not a throughput number. Meanwhile the ``_full`` singlestep config runs with
``thread_workers: 0`` (no workers, no prefetch, reads fully serialized against the GPU).

The bench-vs-production gap is therefore measured between two configs that differ on exactly
the setting that governs I/O overlap. This script measures the setting's actual cost by
running past the prefetch queue at both values.

Read `steady` (not `first`): the first pull includes worker spin-up and zarr metadata paging.
"""

from __future__ import annotations

import argparse
import time

import yaml

from credit.trainers.utils import inject_flat_var_keys, load_dataset, load_dataloader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--batches", type=int, default=12, help="Timed pulls per setting (after 1 untimed).")
    args = ap.parse_args()

    conf = yaml.safe_load(open(args.config))
    inject_flat_var_keys(conf)
    bs = conf["trainer"]["train_batch_size"]
    print(f"config     : {args.config}")
    print(f"batch_size : {bs}   (as authored: thread_workers={conf['trainer']['thread_workers']}, "
          f"prefetch_factor={conf['trainer'].get('prefetch_factor')})\n")

    dataset = load_dataset(conf, is_train=True)

    for workers, prefetch in ((0, 1), (4, 2)):
        conf["trainer"]["thread_workers"] = workers
        conf["trainer"]["prefetch_factor"] = prefetch
        loader = load_dataloader(conf, dataset, rank=0, world_size=1, is_train=True)
        it = iter(loader)

        t0 = time.perf_counter()
        next(it)
        first = time.perf_counter() - t0

        times = []
        for _ in range(args.batches):
            t0 = time.perf_counter()
            next(it)
            times.append(time.perf_counter() - t0)

        # Drop the leading pulls that the prefetch queue could still be covering.
        steady = sum(times[workers * prefetch :]) / max(1, len(times) - workers * prefetch)
        print(f"workers={workers} prefetch={prefetch}:")
        print(f"    first pull : {first:7.1f}s  (spin-up + metadata)")
        print(f"    steady     : {steady:7.1f}s/batch  ({steady / bs:.2f}s/sample)")
        print(f"    all pulls  : {[round(t, 1) for t in times]}\n")

        del it, loader


if __name__ == "__main__":
    main()
