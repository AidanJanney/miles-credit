# Regional Ocean Emulator — Normalization x Rollout-Length x Tendency Experiment Grid

Records the design and exact commands for the first sweep of the gen2 regional MOM6 (carib12)
ocean emulator: **pointwise vs levelwise normalization** x **single-step vs multi-step
training** x **tendency (xi) normalization on vs off**, on **25 vertical levels** selected by a
concrete autoencoder. Builds on the migration described in `OCEAN_MIGRATION_NOTES.md` (read
that first for the gen2 pipeline overview / architecture decisions this experiment inherits
unchanged).

Everything below runs on NSF NCAR Casper, conda env `miles-credit-casper` (editable install of
this repo — code changes here take effect immediately in that env, no reinstall needed). Unlike
earlier revisions of this doc, **no separate `npl`/xgcm environment is needed anywhere anymore**
(see §3 / `OCEAN_MIGRATION_NOTES.md` §5) — everything, including `preprocess_rmom6.py`, runs in
`miles-credit-casper`.

---

## 1. The 8-way grid

2 normalizations x 2 rollout lengths x 2 tendency settings:

| | single-step, tendency ON | single-step, tendency OFF | multi-step, tendency ON | multi-step, tendency OFF |
|---|---|---|---|---|
| **levelwise** | `rmom6_regional_levelwise_singlestep_tendency.yml` | `rmom6_regional_levelwise_singlestep_notendency.yml` | `rmom6_regional_levelwise_multistep_tendency.yml` | `rmom6_regional_levelwise_multistep_notendency.yml` |
| **pointwise** | `rmom6_regional_pointwise_singlestep_tendency.yml` | `rmom6_regional_pointwise_singlestep_notendency.yml` | `rmom6_regional_pointwise_multistep_tendency.yml` | `rmom6_regional_pointwise_multistep_notendency.yml` |

(all under `config/`)

Held fixed across all 8:
- **Vertical levels**: 25 of the native 50 `z1_l` layers, selected by
  `downsample_vgrid_carib12/autoencoder_select.py` (a concrete autoencoder — see §2), not an
  evenly-spaced or manually-chosen subset.
- **Clamping**: unchanged from `config/rmom6_regional.yml` — `ocean_clamp` runs in
  `postblocks.per_step`, but `Reconstruct` detaches `y_processed` before it runs, so clamping
  shapes rollout feedback and saved/rollout output only, **never** the training-step gradient
  (see `OCEAN_MIGRATION_NOTES.md` §2, §6).
- **Data window**: 15-day smoke-test sample, 2000-01-01..2000-01-15 (10 train days / 5 valid
  days).
- **Dynamic forcing**: `taux`, `tauy`, `net_heat_surface`, `runoff` (added this round — see §3).
- **OBC treatment**: a naive edge-overwrite nudge (`ocean_obc_nudge`, see §8) — identical
  across all 8, not itself a grid dimension.

What varies:
- **Normalization type** (levelwise vs pointwise): which preblock/postblock class runs
  (`bridgescaler_transform` vs the new `pointwise_scaler`) — see §4.
- **Rollout length** (`forecast_len: 1` vs `3`): single- vs multi-step truncated-BPTT
  ("pushforward") training. `backprop_on_timestep` is left unset, which defaults to
  `range(1, forecast_len+1)` in `trainer_gen2.py` — every rollout step gets a loss term, but
  gradients don't flow *across* steps (see `OCEAN_MIGRATION_NOTES.md` §6) because `Reconstruct`
  detaches `y_processed`, and the next step's input is built from that detached tensor.
- **Tendency normalization** (xi on vs off): whether the normalization std is pre-scaled by
  `xi` (CREDIT-style residual/tendency normalization, Schreck et al. 2024 arXiv:2411.07814 §3.3:
  `T'' = (T-mu)/(xi*sigma)`) or left as plain `sigma` (`T'' = (T-mu)/sigma`). This is **purely a
  choice of which precomputed stats file the block reads** — `bridgescaler_transform` and
  `pointwise_scaler` never see `xi` themselves; `xi` is folded into `sigma` once by the build
  scripts (`--no-xi` skips it). See §3 and `scripts/build_rmom6_scaler.py` /
  `scripts/build_rmom6_pointwise_scaler.py`.

---

## 2. Vertical level selection (concrete autoencoder)

`downsample_vgrid_carib12/autoencoder_select.py --n-select 25` trains a Concrete Autoencoder
(Balin et al. 2019) over pointwise stats (`explore_statistics_carib12/stats/stats_pointwise_*.nc`)
to pick the 25 of 50 `z1_l` levels most sufficient to reconstruct the full water column
(thickness- and variable-weighted MSE). This is a data-driven pick, not evenly spaced — the loss
weighting (`dz * var_weight`) favors thicker (deeper) layers, so the result skews toward the
mid/deep ocean with only 4 levels in the upper ~10 m:

```
indices : [3, 4, 6, 7, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43]
depths (m): 3.8, 5.1, 7.9, 9.6, 92.3, 130.7, 155.9, 186.1, 222.5, 266.0, 318.1, 380.2, 453.9,
            541.1, 643.6, 763.3, 1062.4, 1245.3, 1452.3, 1684.3, 1941.9, 2225.1, 2533.3, 2865.7, 3220.8
```

Saved to `/glade/derecho/scratch/$USER/rmom6_regional/vgrid_select/`:
- `selected_levels_n25.json` — indices/depths above, in reusable form.
- `autoencoder_n25.pt` — the trained selector + decoder `state_dict`, plus the standardization
  `mu`/`sigma`/`z`/`VARS` needed to run the decoder later. **This is how you decode an emulator's
  25-level prediction back onto the full 50-level column**:
  ```python
  ckpt = torch.load(".../autoencoder_n25.pt")
  selector.load_state_dict(ckpt["selector_state_dict"])
  decoder.load_state_dict(ckpt["decoder_state_dict"])
  # x_standardized: (batch, 50, n_vars) at inference time (only used to build `selector` normally;
  # at deploy time you instead already have the 25 selected levels' values from the emulator)
  selected = ...        # (batch, 25, n_vars) — the emulator's predicted values at the 25 levels,
                         # standardized the same way (ckpt["mu"], ckpt["sigma"])
  recon_standardized = decoder(selected)
  recon_physical = recon_standardized * ckpt["sigma"] + ckpt["mu"]   # -> full 50-level profile
  ```

This 25-level list is what feeds `--levels` in every preprocessing/scaler-building command
below — it must stay identical across every step (prognostic store, both scalers, OBC level
selection) or channel order/count will silently misalign (levels are matched by *position* in
the list you pass, independently nearest-matched against each file's own `z1_l` coordinate).

---

## 3. New/changed files

| File | What it is |
|---|---|
| `downsample_vgrid_carib12/autoencoder_select.py` | *(modified)* now saves the decoder checkpoint + selected-levels JSON. |
| `credit/preblock/pointwise_stats.py` | Shared helper: loads a torch-saved `{var_key: {"mu","sigma"}}` stats dict and applies the elementwise (de)standardization against a `(B, nz, T, H, W)` tensor. |
| `credit/preblock/pointwise_scaler.py` | Preblock `pointwise_scaler` — the pointwise analogue of `bridgescaler_transform`, registered in `PREBLOCK_REGISTRY`. |
| `credit/postblock/pointwise_scaler.py` | Postblock, same config interface, registered in `POSTBLOCK_REGISTRY`. |
| `credit/postblock/ocean_obc_nudge.py` | New postblock `ocean_obc_nudge` — naive OBC boundary-ring overwrite (see §8). |
| `credit/trainers/rollout_utils.py` | *(modified)* `assemble_rollout_batch` now forwards `curr_batch["metadata"]` through (was silently dropped after rollout step 0 — see §8). |
| `scripts/build_rmom6_scaler.py` | *(modified)* folds `stats_xi.nc` into std (levelwise), now with a `--no-xi` flag to skip it (plain std). |
| `scripts/build_rmom6_pointwise_scaler.py` | Pointwise + xi (or plain, with `--no-xi`) stats builder, from `stats_pointwise_<var>.nc` x `stats_xi.nc`. Includes a C-to-A grid regrid for uo/vo. |
| `scripts/preprocess_rmom6.py` | *(modified)* Arakawa C-to-A regrid switched from `xgcm` to a plain numpy/xarray `center_average()` (verified bit-exact — see §3's ⚠️ below); `FORCING_SPEC` gained `runoff` (`friver`). No longer needs the `npl` env. |
| `config/rmom6_regional.yml` | *(modified)* base template: `runoff` added to `dynamic_forcing`, `input_only_channels` 5→6. |
| `config/rmom6_regional_{levelwise,pointwise}_{singlestep,multistep}_{tendency,notendency}.yml` | The 8 experiment configs. |
| `scripts/submit_rmom6_<variant>.sh` | 8 PBS submission wrapper scripts (see §10). |

⚠️ **Two things found and fixed while smoke-testing:**
1. `stats_pointwise_uo.nc`/`stats_pointwise_vo.nc` were computed on the **raw archive's native
   Arakawa-C face grid** (`uo` on `xq`, 760 points; `vo` on `yq`, 458 points), not the A-grid
   (`xh`/`yh`, 759/457) the dataset actually reads. Fixed in `build_rmom6_pointwise_scaler.py`
   with a center-average regrid of the mean/std fields (`mean` is exact this way — linear,
   commutes with time-averaging; `std` is a documented approximation, see
   `OCEAN_MIGRATION_NOTES.md` §5).
2. `preprocess_rmom6.py`'s `xgcm`-based regrid required a second conda env (`npl`). Replaced
   with the identical formula by hand (`center_average()`), verified **bit-exact** against the
   old xgcm output for uo/vo/thetao/so/SSH (max abs diff `0.0`) before switching over — the
   whole pipeline now runs in one env.

---

## 4. Why a new preblock/postblock was needed for pointwise

`bridgescaler_transform` (existing) reads a `DStandardScalerTensor` whose `mean_x_`/`var_x_` are
strictly 1D, one value per channel (level) — `reshape_to_channels_first` inserts a singleton dim
at every axis except the channel axis, so it can only ever broadcast a stat uniformly across
H/W. It has no way to represent a mean/std that also varies spatially, which pointwise
normalization requires by definition. Levelwise fits this class unchanged (`sigma` and `xi` are
already 1D-per-level, whether or not `xi` is folded in); pointwise does not, hence
`pointwise_scaler`.

`pointwise_scaler`'s stats file (torch-saved) has the form:
```
{var_key: {"mu": tensor(nz, H, W), "sigma": tensor(nz, H, W)}, ...}
```
`sigma` has `xi` folded in already (`sigma_eff = xi[level] * sigma_pointwise[level, y, x]`) for
the `_tendency` configs, or is plain `sigma_pointwise` for the `_notendency` configs — same
build script, `--no-xi` toggles it. The blocks themselves do nothing but `(x - mu) / sigma` /
its inverse, broadcast against the batch's `(B, nz, T, H, W)` tensor. Land/seafloor points (NaN
or non-positive std in the source stats) are filled `mu=0, sigma=1` at build time.

---

## 5. Commands to reproduce the smoke test from scratch

All in `miles-credit-casper` — no other env needed (see §3).

```bash
module load conda
conda activate miles-credit-casper

# 1. Select 25 vertical levels (writes selected_levels_n25.json + autoencoder_n25.pt)
cd /glade/work/ajanney/RegionalEmulation_v2/downsample_vgrid_carib12
python3 autoencoder_select.py --n-select 25
cd /glade/work/ajanney/RegionalEmulation_v2/miles-credit-regional

LEVELS="3.8,5.1,7.9,9.6,92.3,130.7,155.9,186.1,222.5,266.0,318.1,380.2,453.9,541.1,643.6,763.3,1062.4,1245.3,1452.3,1684.3,1941.9,2225.1,2533.3,2865.7,3220.8"
OUT=/glade/derecho/scratch/$USER/rmom6_regional/preprocessed
SCALER=/glade/derecho/scratch/$USER/rmom6_regional/scaler

# 2. Re-preprocess the 15-day sample restricted to those 25 levels (prognostic + forcing incl. runoff)
python -u scripts/preprocess_rmom6.py \
  --start-date 2000-01-01 --end-date 2000-01-16 \
  --levels "$LEVELS" --sample-label sample25 --out-dir "$OUT" --overwrite

# 3. OBC boundary zarrs (needed by ocean_obc_nudge -- see §8)
python -u scripts/preprocess_obcs.py \
  --start-date 2000-01-01 --end-date 2000-01-16 --out-dir "$OUT" --overwrite

# 4. Build all 4 scaler artifacts: {levelwise,pointwise} x {tendency,notendency}
python -u scripts/build_rmom6_scaler.py --levels "$LEVELS" \
  --out "$SCALER/ocean_bridgescaler_levelwise_xi_n25.json"
python -u scripts/build_rmom6_scaler.py --levels "$LEVELS" --no-xi \
  --out "$SCALER/ocean_bridgescaler_levelwise_notendency_n25.json"
python -u scripts/build_rmom6_pointwise_scaler.py --levels "$LEVELS" \
  --out "$SCALER/ocean_pointwise_xi_n25.pt"
python -u scripts/build_rmom6_pointwise_scaler.py --levels "$LEVELS" --no-xi \
  --out "$SCALER/ocean_pointwise_notendency_n25.pt"

# 5. Train each of the 8 configs (single A100 -> run sequentially)
for cfg in config/rmom6_regional_{levelwise,pointwise}_{singlestep,multistep}_{tendency,notendency}.yml; do
  credit train -c "$cfg"
done
```

Or submit each as a PBS batch job instead of step 5 — see §10.

---

## 6. Smoke-test verification status — all 8 configs pass

Each config was checked with a manual dataset -> preblocks -> postblocks pass (mirrors
`OCEAN_MIGRATION_NOTES.md`'s T1-T3): real 25-level sample data in, normalized `x` tensor out (no
NaN, channel count `4*25 + 1 = 101` prognostic + `4+2=6` forcing/static = **107**, matching
`model.channels`/`levels`/`surface_channels`/`input_only_channels`), a synthetic `y_pred`
through the full postblock chain out (denormalized values land within `ocean_clamp`'s physical
bounds, no NaN, `runoff` present in the dataset's own output).

All 8 configs were then run end-to-end with `credit train -c <config>.yml` (5 epochs, single
A100, batch_size 1, `activation_checkpoint: True`) on the 15-day/25-level sample. All 8
completed cleanly (exit 0), GPU memory ~3 GB peak / 81 GB available (4%), loss finite throughout,
checkpoints saved each epoch. Loss magnitudes aren't comparable across configs (different
normalization scales; multistep sums loss over 3 steps vs 1) — the bar is "runs without error,
loss finite," which all 8 satisfy.

**OBC nudge correctness was verified directly** (not just "doesn't crash"): a standalone 2-step
rollout (real dataset -> preblocks -> fake model output -> postblocks -> `assemble_rollout_batch`
-> preblocks again) confirmed that (a) `batch_dict["metadata"]["target"]["rMOM6"]["datetime"]`
is now present and correct at step 2 (previously would've been silently missing), and (b) the
north-edge `thetao` values after `ocean_obc_nudge` match `obc_north_sample.zarr`'s value at the
nearest matching timestamp/levels exactly (4/5 sampled points bit-matched; the 5th is the
north-west corner, which — as documented in the postblock's own docstring — gets overwritten
twice, once by the north edge and once by the west edge; west's OBC value at that corner is
independently confirmed to be `0.0`, explaining the difference. Not a bug.).

---

## 7. Graduating to the full run

Once an 8-way smoke test looks right (loss finite/decreasing, no NaN, clamp bounds respected in
rollout), for each config:

1. **Re-preprocess the full 20-year archive at 25 levels** (no `--start-date`/`--end-date`,
   default `--years 2000-2019`):
   ```bash
   python -u scripts/preprocess_rmom6.py --levels "$LEVELS" \
     --out-dir /glade/derecho/scratch/$USER/rmom6_regional/preprocessed
   ```
   This writes annual `rmom6_prognostic_<year>.zarr`/`rmom6_forcing_<year>.zarr` stores (the
   `%Y`-patterned naming `LocalDataset` globs across years automatically). Also re-run
   `preprocess_obcs.py` without a date range for the full OBC record.
2. **Statistics don't need to be recomputed** — `stats_levelwise.nc`/`stats_pointwise_*.nc`/
   `stats_xi.nc` are already fit over the full training period, not the 15-day sample; the
   scaler-building commands in §5 step 4 are unchanged (same `--levels`, same output paths).
3. **In each config**, change the `..._sample25.zarr` / `obc_*_sample.zarr` paths to the
   `%Y`-patterned / non-`_sample` form; widen `start_datetime`/`end_datetime` to the real
   train/valid split you want across 2000-2019 (e.g. train on 2000-2017, validate on 2018-2019).
4. **Widen `pbs:`** to a multi-epoch production job (this smoke test's `pbs.walltime: 00:30:00`
   / single-GPU Casper sizing won't hold for 20 years of data) — see §9/§10 for multi-GPU and
   submission.

---

## 8. Open boundary conditions (OBCs) — naive nudge, dedicated pathway still needed

Until this round, OBCs were **not connected to the pipeline at all**: `scripts/preprocess_obcs.py`
produces `obc_{direction}[_sample].zarr` (boundary-strip zarrs — `(time, z1_l, xh_or_yh)`, one
edge each), but nothing read them, for any of the 8 configs, single-step or multi-step.

**What was added — `ocean_obc_nudge` (naive placeholder).** Every `postblocks.per_step` pass
(after `ocean_wet_mask`, before `post_rollout`), it overwrites the single-pixel-wide outer edge
ring of `thetao`/`so`/`uo`/`vo`/`SSH` in `y_processed` with the prescribed OBC value at the
current step's real timestamp, nearest-matched against each `obc_{direction}.zarr`'s own `time`
coordinate (and against `--levels` for the 3D vars). Config block (identical in all 8 configs):

```yaml
postblocks:
  per_step:
    ...
    wet_mask: { ... }
    obc_nudge:
      type: ocean_obc_nudge
      args:
        obc_paths: { north: ..., south: ..., east: ..., west: ... }
        variables: ["rMOM6/prognostic/3d/thetao", ..., "rMOM6/prognostic/2d/SSH"]
        levels: [3.8, 5.1, ...]   # must match model.levels
  post_rollout: { ... }
```

**A real bug had to be fixed first**: `credit/trainers/rollout_utils.py`'s
`assemble_rollout_batch` used to return `{"input":..., "target":...}` with no `"metadata"` key,
so `ConcatToTensor` (`credit/preblock/concat.py:93-100`) never populated
`metadata["target"][source]["datetime"]` after rollout step 0 — the timestamp was silently gone
for the rest of the rollout. Fixed by forwarding `curr_batch.get("metadata")` through, the same
way `"target"` already was. This affects **all** gen2 rollout/multi-step training, not just
ocean — any future postblock needing "what time is it" now has a real answer at every step.

**This is not a real fix, just a stopgap** (see `OCEAN_MIGRATION_NOTES.md` §5 for the full
writeup):
- No relaxation timescale / smooth sponge-layer blend into the interior — a real regional-model
  OBC treatment nudges over a boundary *zone*, not one pixel.
- The model itself never sees the OBC as an input — the overwrite only shapes what feeds the
  *next* step, not the current step's forward pass or loss.
- Corner pixels (where two edges meet) get overwritten twice; whichever direction is processed
  last in `obc_paths` wins. Documented, not fixed (harmless for a placeholder).

**Real fix — dedicated OBC input pathway (future work, not built yet).** Two designs sketched:
1. Scatter each boundary strip into a full-domain-shaped field (zero off the boundary) at
   preprocessing time so it rides in as an ordinary `dynamic_forcing` variable through the
   existing `MultiSourceDataset` → `ConcatToTensor` pipeline. Simplest, no model changes, but
   wastes most of the channel on padding.
2. A genuinely separate OBC source/preblock that keeps the boundary strips in their native 1D
   shape and feeds them into the encoder directly (a boundary embedding, or an additive
   correction term near the domain edge). More architecture work, doesn't waste channels, and
   lets the model actually condition on the boundary forcing.

---

## 9. Multi-GPU

Yes — every config already sets `trainer.parallelism.data: ddp` (`tensor: 1`, `domain: 1`, so
only data-parallel replication is engaged, not tensor- or domain-parallelism). Launch via
`credit submit` (see §10), which generates a PBS script that runs
`torchrun --standalone --nnodes=1 --nproc-per-node=$NGPUS train_gen2.py -c $CONFIG` on Casper —
no config changes needed, just `--gpus N` (verified with `--dry-run` up to `N=4`, a Casper A100
node's typical GPU count). DDP splits each epoch's samples across the `N` ranks and all-reduces
gradients; no manual sharding needed.

**Caveat for the current smoke test**: the 15-day sample only has ~9 (singlestep) / ~7
(multistep) training samples total, so 4 GPUs mostly just divides an already-tiny epoch further
— multi-GPU only pays off once you've graduated to the full 20-year run (§7), where DDP gives a
close-to-linear speedup up to the node's GPU count.

---

## 10. PBS submission scripts

`scripts/submit_rmom6_<variant>.sh` (8 total, one per config) wrap `credit submit`:
```bash
./scripts/submit_rmom6_levelwise_singlestep_tendency.sh                 # 1 GPU, resources from the config's pbs: block
./scripts/submit_rmom6_pointwise_multistep_notendency.sh --gpus 4        # 4-GPU DDP run
./scripts/submit_rmom6_levelwise_multistep_tendency.sh --dry-run         # print the PBS script, don't submit
```
Each currently points at its 15-day **SAMPLE** config — swap in a full-run config (§7) before
submitting for real. `credit submit` reads `walltime`/`ncpus`/`ngpus`/`mem`/`queue`/`conda` from
the target config's own `pbs:` block by default (verified via `--dry-run`); any CLI flag
(`--gpus`, `--walltime`, ...) overrides that. It also auto-chains jobs via PBS `afterok`
dependencies to cover the config's full `trainer.epochs` in `trainer.num_epoch`-sized segments
(e.g. 20 epochs ÷ 5/job = 4 chained jobs for the current smoke-test configs).

---

## 11. Pathway reference — what calls what, for each workflow

### A. Vertical level selection (once)

| Step | Script | Reads | Produces / calls |
|---|---|---|---|
| 1 | `downsample_vgrid_carib12/autoencoder_select.py --n-select 25` | `explore_statistics_carib12/stats/stats_pointwise_{thetao,so,uo,vo,SSH}.nc` | Trains `ConcreteSelect` + `Decoder` (pure PyTorch, no CREDIT code) → `vgrid_select/selected_levels_n25.json`, `vgrid_select/autoencoder_n25.pt` |

### B. Data preprocessing (prognostic + forcing + OBCs)

| Step | Script | Reads | Produces / calls |
|---|---|---|---|
| 1 | `scripts/preprocess_rmom6.py` | Raw archive `*.mom6.h.{glorys,sfc,atm}.YYYY-MM.nc`; `--levels` from step A | `center_average()` (in-file, no xgcm) regrids uo/vo/taux/tauy C→A → `rmom6_prognostic_sample25.zarr`, `rmom6_forcing_sample25.zarr` |
| 2 | `scripts/preprocess_obcs.py` | `<obc-dir>/forcing_obc_segment_00{1..4}.nc` | `_decode_obc_time()` (calendar fix), per-segment center-point extraction → `obc_{north,south,east,west}_sample.zarr` |

### C. Scaler / stats building (once per normalization x tendency combo)

| Step | Script | Reads | Produces / calls |
|---|---|---|---|
| 1a | `scripts/build_rmom6_scaler.py [--no-xi]` | `explore_statistics_carib12/stats/{stats_levelwise,stats_global,stats_xi}.nc` | `_make_scaler()` → `bridgescaler.distributed_tensor.DStandardScalerTensor` (populated directly, not fit) → `bridgescaler.save_scaler_dict()` → `ocean_bridgescaler_{levelwise_xi,levelwise_notendency}_n25.json` |
| 1b | `scripts/build_rmom6_pointwise_scaler.py [--no-xi]` | `stats_pointwise_{thetao,so,uo,vo,SSH}.nc`, `stats_xi.nc` | `_center_average()` (uo/vo C→A regrid) + `_fill_stat()` (land no-op) → `torch.save()` → `ocean_pointwise_{xi,notendency}_n25.pt` |

### D. Training (one config, `credit train -c <config>.yml`)

| Step | Module | Reads | Calls into |
|---|---|---|---|
| 1 | `credit/applications/train_gen2.py` (entry point) | The config YAML | Builds dataset, model, trainer per config `data:`/`model:`/`trainer:` blocks |
| 2 | `credit/datasets/multi_source.py: MultiSourceDataset` | `data.source.rMOM6` block | Routes to `credit/datasets/rmom6.py: RegionalMOM6Dataset` (`_SOURCE_REGISTRY["RMOM6"]`) per sample |
| 3 | `credit/datasets/rmom6.py: RegionalMOM6Dataset` | `rmom6_prognostic_sample25.zarr`, `rmom6_forcing_sample25.zarr`, the static `.nc` | Returns nested `{"input": {...}, "target": {...}, "metadata": {...}}` per `(datetime, i)` |
| 4 | `credit/models/__init__.py` (`_MODEL_REGISTRY["wxformer"]`) | `model:` block | `credit/models/wxformer/crossformer.py: CrossFormer` |
| 5 | `credit/trainers/trainer_gen2.py: TrainerERA5Gen2.train_one_epoch` | `trainer:`, `data.forecast_len` | Per rollout step `t`: `build_preblocks`/`apply_preblocks` (`credit/preblock/__init__.py`) → model forward → `build_postblocks`/`apply_postblocks` (`credit/postblock/__init__.py`) → (if `t < forecast_len`) `credit/trainers/rollout_utils.py: assemble_rollout_batch` |
| 6a | preblocks (`preblocks.per_step`, config order) | `normalize` → `bridgescaler_transform` (`credit/preblock/scaler.py`) **or** `pointwise_scaler` (`credit/preblock/pointwise_scaler.py` → `pointwise_stats.py`), reading the scaler/stats file from step C | `fill_land` (`credit/preblock/fill_values.py`) → `concat` (`credit/preblock/concat.py`) builds `x`/`y`/`metadata` |
| 6b | postblocks (`postblocks.per_step`, config order) | `reconstruct` (`credit/postblock/reconstruct.py`, needs `metadata["target"]["_channel_map"]`) | `denorm` (inverse of 6a) → `clamp` (`credit/postblock/ocean_clamp.py`) → `wet_mask` (`credit/postblock/ocean_wet_mask.py`, reads the static file + `level_source_path`) → `obc_nudge` (`credit/postblock/ocean_obc_nudge.py`, reads `obc_*.zarr` + `metadata["target"][source]["datetime"]`) |
| 7 | postblocks (`postblocks.post_rollout`, once) | `density` (`credit/postblock/ocean_density.py`) | Diagnoses `rho` from the final step's `thetao`/`so` |

### E. Rollout / inference (`credit rollout -c <config>.yml`, or the `inference:` block during `credit submit --mode rollout`)

| Step | Module | Reads | Calls into |
|---|---|---|---|
| 1 | `credit/applications/rollout_gen2.py` | `inference:` block | `credit/trainers/rollout_utils.py: run_forecast` (IC load + autoregressive loop) |
| 2 | Step 0 (IC) | Dataset's raw batch at `inference.single_forecast.start_datetime` | `ic_preblocks` (`ic_only` phase) → `step_preblocks` (`per_step` phase) → `full_data_dict["x"]` |
| 3 | Steps 1..n | Model forward → `apply_postblocks(step_postblocks, ...)` (same chain as D.6b) → `save_output_fn` writes `y_processed` (`credit/output_gen2.py: ForecastWriter`) → if not last step, `assemble_rollout_batch` + `step_preblocks` again | Same `ocean_obc_nudge`/`ocean_wet_mask`/etc. as training — this is why the `assemble_rollout_batch` metadata fix (§8) matters equally here |
| 4 | End | `apply_postblocks(rollout_postblocks, ...)` (post_rollout, once) | `ocean_density` |

Every workflow above ultimately bottoms out in the same two registries —
`credit/preblock/__init__.py: PREBLOCK_REGISTRY` and `credit/postblock/__init__.py:
POSTBLOCK_REGISTRY` — mapping each config `type:` string to the class actually instantiated; that
is the single place to look up "what does this block's code actually do" for any step in the
tables above.
