# Regional MOM6 training: unexplained ~10x slowdown vs. isolated benchmark

## TL;DR

Real `credit train` runs (via `train_gen2.py` / `TrainerERA5Gen2`) run **~10x slower per
iteration** than an isolated script that builds the same model and calls `model(x)` directly,
under otherwise-matched settings (same weights, same batch size, same GPU, same conda env).
This affects **both** singlestep and multistep configs, is **not** caused by chained rollout
depth (per-step timing is flat, not growing), and is **not** explained by any of the causes
ruled out below. It doesn't block correctness — runs complete successfully, just slower than
expected — but it's the reason `batches_per_epoch` has to be capped low in the `_full` configs,
which in turn is why 20-epoch full-scale runs are undertrained (see
`config/rmom6_regional_levelwise_singlestep_tendency_full.yml`'s own comment: "a full 52-iter
epoch x10 still too long").

A **separate, unrelated bug** was also found: multi-GPU DDP crashes with a real NCCL error.
See "Bug 2" below.

## Reproducing

- Isolated (fast) baseline: `scripts/benchmark_rmom6.py -c config/rmom6_bench_singlestep.yml --iters 3`
  (and `rmom6_bench_multistep.yml`) — measures ~4.1s/iter (singlestep) / 30.5s/iter (multistep)
  at batch_size=16, full 2000-2017 data.
- Real trainer (slow): submit any `_full` config via `credit submit`, or add temporary timing
  instrumentation to `TrainerERA5Gen2.train_one_epoch`'s inner loop (see "How I instrumented it"
  below) — measures ~45.8s/iter (singlestep) / ~150-245s/iter (multistep) at the same settings.

Concretely, on the night of 2026-07-19/20, `rmom6_regional_levelwise_singlestep_tendency_full.yml`
resumed-from-scratch and ran 20 epochs (10 batches/epoch, batch_size=16, full 2000-2017 window)
in **2h43m wall-clock** — that's ~45.8s/iter, vs. ~4.1s/iter in the isolated benchmark for
identical settings.

## What's been ruled out (with evidence)

| Candidate | How tested | Result |
|---|---|---|
| Cross-cluster / slow storage I/O | Solo zarr-read timing test on Casper | 0.5s/timestep raw read — fast |
| Node/GPU contention (other jobs on the shared node) | Killed co-located jobs, reran solo | No change in timing |
| FSDP2/DDP wrapping differences vs. the benchmark | Traced `distributed_model_wrapper_gen2`: at `dp_size<=1`, FSDP2 is skipped and activation checkpointing is applied via the *same* function the benchmark calls | Code paths are identical for single-GPU |
| Corrupted/extreme trained weights (denormal floats etc.) | Isolated script: forward pass with fresh-init weights vs. checkpoint weights, both under `no_grad` | Identical speed (0.63s each); checkpoint weights have 0 non-finite values, max abs 1.0 |
| Activation checkpointing overhead under autograd | Isolated script: single forward+backward, checkpointing on vs. off | 2.6s vs. 2.4s — negligible difference |
| `torch.backends.cudnn.benchmark` (set `True` in `train_gen2.py:160`, `False` in the benchmark script) | Made it overridable via `CREDIT_DEBUG_NO_CUDNN_BENCHMARK=1`, reran the real trainer with it forced `False` | No change (18.7s vs. 20.4s per step) |
| GPU memory pressure / allocator fragmentation | Checked live `nvidia-smi` during a slow run | Only 14% memory utilized — not pressure-bound |
| Growing autograd graph across the 10 chained rollout steps (multistep only) | Per-t timing instrumentation, printed each of the 10 steps individually | Flat ~16-20s/step from t=1 onward, not growing — rules out chain-depth theory |
| GPU hardware class difference (Casper vs. Derecho) | `nvidia-smi` on both: A100-SXM4, compute cap 8.0, no MIG on either | Same GPU class |
| Model/trainer code being a fork-specific regression | Diffed `credit/trainers/trainer_gen2.py`, `credit/models/wxformer/crossformer.py`, and `credit/trainers/rollout_utils.py` against `/glade/work/ajanney/miles-credit-v2` (the reference repo) | `forward()` methods are byte-identical; trainer differences are all TP/domain-parallel/grad-accum additions that are no-ops for this single-GPU, non-domain-parallel config |
| `LatWeightedMetrics` computing accuracy/MAE (~101-variable Python loop, ~300+ forced GPU syncs/iteration) | Vectorized it (see "Real fix" below), confirmed via production instrumentation it dropped to 0.10s/iteration | **Real bug, fixed — but fixing it alone did not close the timing gap** (iteration time was unchanged before/after in the real trainer) |

**Bottom line:** every isolated/controlled variable that plausibly explains a large,
per-iteration-independent slowdown has been eliminated. The gap is real, reproducible,
and localized to "running inside the full `train_one_epoch` context" vs. "a clean script
calling `model(x)` directly" — but the specific mechanism was not found.

## Ideas not yet tried

- **Systematic PyTorch profiler** (`torch.profiler`) around one real iteration, comparing the
  op-level trace against the isolated script's trace. All the ad-hoc timing done so far used
  `time.perf_counter()` + `torch.cuda.synchronize()` around coarse phases — a real profiler
  would show which specific kernels/ops are slow, rather than requiring more guess-and-test
  cycles.
- **`torch.autograd.profiler.record_function`** markers plus checking CPU-side Python overhead
  specifically (e.g., is the trainer's `dl = cycle(trainloader)` / dataloader worker interaction
  somehow serializing with the GPU work in a way a clean script's dataloader-free forward call
  wouldn't?).
- Check whether the **preflight batch** (`credit/trainers/preflight.py`, "fetching first
  training batch") leaves some global torch/CUDA state (e.g., a stream, a memory pool, a
  `torch.set_num_threads` call) in a different configuration than a fresh process.
- Check **`OMP_NUM_THREADS`/thread affinity**: the real trainer runs under `torchrun` with a
  DataLoader using `thread_workers: 4` background processes; a clean script has neither. If
  something (torchrun's elastic agent, or DataLoader worker processes) is pinning CPU cores
  in a way that starves the main process during CUDA kernel launches, that could produce
  exactly this "100% GPU util but only ~50% power draw" signature seen live via `nvidia-smi`
  (recorded during one of the slow runs; not yet compared against the isolated script's numbers).
- Diff **`credit/trainers/preflight.py`** itself against upstream — it's new/regional-specific
  and untested tonight; the GPU memory check it prints happens right before the first real
  iteration, so if it does anything expensive to torch/CUDA global state, this is where.

## Bug 2 (separate): multi-GPU DDP crashes

`credit submit --cluster casper -c config/rmom6_regional_levelwise_singlestep_tendency_full.yml`
(4 GPUs, as authored) failed with:

```
torch.distributed.DistBackendError: NCCL error in: .../ProcessGroupNCCL.cpp:3087,
internal error - please report this issue to the NCCL developers, NCCL version 2.21.5
ncclInternalError: Internal check failed.
```

Not investigated further tonight — worked around by forcing `--gpus 1 --nodes 1` for the
overnight run. Given `parallelism.data: ddp` in the `_full` configs and `pbs.ngpus: 4` /
2-node defaults, **any full-scale run submitted without an explicit `--gpus 1` override will
hit this.** Worth checking NCCL/PyTorch version compatibility on whichever cluster this
reproduces on (only tested via `credit submit --cluster casper`, targeting Derecho GPUs across
2 nodes — see the generated PBS script's `mpiexec -n 8 --ppn 4` launch for the exact repro
command).

## Real fix applied and verified: `credit/metrics.py`

`LatWeightedMetrics.__call__` had a Python `for` loop over every model output variable
(~101 for this config: 4 3D vars × 25 levels + surface + diagnostic), each iteration doing
several unbatched GPU reductions and — critically — the aggregation step called `.cpu().item()`
per-variable for the `acc` metric (and `.cpu()` for rmse/mse/mae), forcing 300+ GPU
synchronizations every training iteration, independent of batch size or `forecast_len`.

Rewrote it to batch the reductions across the channel dimension and do exactly one
`.cpu()` transfer at the end. **Verified numerically identical to the original** via a
standalone equivalence test (5 cases: 4D/5D input, with/without latitude weights,
with/without ensembling; max abs error ~2e-7, pure floating-point reduction-order noise).
Confirmed via real production instrumentation that it dropped from a real cost to 0.10s of
a 245s iteration.

**This is a legitimate, independent fix** (uncommitted — `git diff credit/metrics.py`) worth
keeping regardless of the broader mystery. It just isn't sufficient on its own to explain or
fix the ~10x gap.

## How I instrumented it (for whoever picks this up)

Added temporary per-phase (and later per-rollout-step) timers directly inside
`TrainerERA5Gen2.train_one_epoch`, guarded by `if os.environ.get("CREDIT_DEBUG_TIMING")`, using
`torch.cuda.synchronize()` + `time.perf_counter()` around each phase (`data`, `preblocks`,
`forward`, `postblocks`, `backward`, `optimizer`, `metrics`). Reverted after each test via
`git checkout -- credit/trainers/trainer_gen2.py` since it's not meant to ship. Recommend
reintroducing this (or a `torch.profiler`-based version) as the next step, run against a
**warmed `save_loc`** (i.e. an experiment directory that already has a valid checkpoint at the
*same* `train_batch_size` the test will use — a fresh/mismatched directory triggers an
unrelated pre-existing bug, see "Known unrelated bug" below).

## Known unrelated bug: fresh/batch-size-mismatched `save_loc` crashes preblocks

Separately discovered: starting training against a **brand-new `save_loc`**, or an existing
`save_loc` whose checkpoint was trained at a **different `train_batch_size`** than the current
run, crashes with:

```
File ".../credit/preblock/_utils.py", line 28, in _parse_variable_selection
    for source in state_dict.get(data_type, {}).values():
AttributeError: 'Tensor' object has no attribute 'get'
```

Root cause not identified — best guess is something in `ic_preblocks` (static regrid?) or
`fill_values.py`'s lazy variable-selection caching (`self.variables` starts as `[]` and is
resolved on first call) behaves differently on a true first-time-computation path. Workaround:
always test/resume against a `save_loc` with a checkpoint saved at the **same batch size**
you're about to run.

## Relevant files

- `credit/trainers/trainer_gen2.py` — `TrainerERA5Gen2.train_one_epoch`, the real training loop
- `credit/applications/train_gen2.py` — entrypoint; sets `cudnn.benchmark`, launches via torchrun
- `credit/metrics.py` — the fixed `LatWeightedMetrics`
- `credit/trainers/preflight.py` — untested lead, see "Ideas not yet tried"
- `scripts/benchmark_rmom6.py` — the isolated (fast) benchmark script, also had its own
  unrelated bug fixed tonight (shared `cycle(loader)` iterator across STAGE 1/2/3 went out of
  phase for `forecast_len > 1`, now each stage gets its own iterator)
- `config/rmom6_regional_levelwise_singlestep_tendency_full.yml` — the config used for the
  overnight validation run; its own comments (`batches_per_epoch: 10 # ... a full 52-iter
  epoch x10 still too long`) show this slowdown has been silently budgeted around before,
  not newly discovered tonight
