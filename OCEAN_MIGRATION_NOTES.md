# Regional MOM6 Ocean Emulator — gen2 Migration: Implementation & Testing Notes

Implements the migration plan on branch `regional_emulation_v2`. This document
records **what was built**, **what must still be done offline**, and a
**testing checklist** to run once environments are set up. Nothing here has been
executed yet — no test environment was available at implementation time.

---

## 1. What was implemented (files touched)

### New files
| File | Purpose |
|---|---|
| `credit/datasets/rmom6.py` | `RegionalMOM6Dataset` — gen2 ocean dataset (thin `LocalDataset` subclass) |
| `credit/postblock/ocean_wet_mask.py` | `OceanWetMask` — zero land points from static `wet`/`deptho`/`zl` |
| `credit/postblock/ocean_clamp.py` | `OceanClamp` — clamp variables to physical bounds |
| `credit/postblock/ocean_density.py` | `OceanDensity` — diagnose seawater potential density (linear EOS) |
| `config/rmom6_regional.yml` | End-to-end example config |

Normalization uses the shipped `bridgescaler_transform` (pre + post) — no custom
normalization block. (An earlier `mean_std_denorm.py` was added then removed once
bridgescaler was chosen; see §2.)

### Edited files (surgical, additive only)
| File | Change |
|---|---|
| `credit/datasets/multi_source.py` | `_SOURCE_REGISTRY["RMOM6"] = ("credit.datasets.rmom6", "RegionalMOM6Dataset")` |
| `credit/postblock/__init__.py` | Import + register `ocean_wet_mask`, `ocean_clamp`, `ocean_density`, `mean_std_denormalize` |

**Not touched:** `wet_mask_samudra.py` and the OM4/Samudra `Ocean_*_Batcher` path
are left as-is (they serve the global-ocean Samudra model, a separate pipeline).

---

## 2. Architecture decisions (and why)

- **Dataset reuses `LocalDataset`.** The CREDIT input `.nc` files are ordinary
  gridded NetCDF (`time, zl, yh, xh`), so the generic loader already handles
  per-variable extraction, level selection, cftime, and the nested output
  schema. `RegionalMOM6Dataset` overrides only `_extract_field`/`_select_time`
  for two ocean quirks: MOM6's **singleton `time` dim on the static file**
  (`isel(time=0)`), and a **`rename` map** so config can use canonical names
  (`thetao`/`so`) while files store `temp`/`salt`.

- **OBCs enter as `dynamic_forcing` (plan Option A).** The open boundary
  conditions are already prepared on the **same grid as the model state** (grid
  and calendar alignment done upstream by the data owner), so they need no
  regridding or special handling in the loader — list them as ordinary
  `dynamic_forcing` variables and they flow through like any other forcing. A
  dedicated **OBC layer/source** (keeping boundaries as a separate input the
  model treats specially) is a possible **future** direction, not needed now.

- **Postblocks operate on `y_processed`.** `trainer_gen2` passes `full_data_dict`
  through `apply_postblocks`; `Reconstruct` splits `y_pred` into
  `y_processed[source][var_key] = (B, n_levels, n_time, H, W)`. All ocean
  postblocks follow the same nested-dict contract as `BridgeScalerTransform`
  (postblock/scaler.py) — **not** the `data_dict[data_type][dataset_name]`
  convention used by `mslp.py`/`geopotential.py`.

- **⚠️ Postblocks are DETACHED from the training gradient.** `Reconstruct`
  calls `.detach()` on `y_processed`. Therefore `ocean_clamp`, `ocean_wet_mask`,
  and `ocean_density` shape **rollout feed-back** (via `_gather_for_next_step`)
  and **written/inference outputs**, but do **not** influence the single-step MSE
  loss gradient. If you want physical constraints to shape training gradients,
  that is a different integration point (apply to `y_pred` before detach) and is
  out of scope here.

- **Normalization = bridgescaler (`standard` scaler).** bridgescaler's
  `standard` scaler is per-variable z-score (mean/std) — the same statistic the
  old ocean pipeline's `*_mean.nc`/`*_std.nc` files encode — but as a fitted
  object with a built-in `inverse_transform` and one shared artifact for the
  whole pre/post round-trip. Preblock `bridgescaler_transform` (method:
  `transform`) + postblock `bridgescaler_transform` (method: `inverse_transform`)
  read the **same** `scaler.json`. Generate it once with `credit preprocess`
  over the ocean data. This replaced an earlier bespoke `mean_std_denormalize`
  postblock (removed), which is unnecessary now that bridgescaler owns the
  inverse. The shipped `era5_normalizer` preblock (raw mean/std NC) still exists
  if you ever want to consume the old NC files directly, but it has **no**
  matching inverse postblock — which is exactly why bridgescaler is preferred.

---

## 3. Offline preprocessing still required (before any run)

These build the inputs the pipeline consumes; they are **not** part of CREDIT
and must be done first. Reference scripts live in
`/glade/work/ajanney/Regional_Ocean_Emulation/archive/DataLoader_Input/` and
`archive/Thesis_Archive/`.

1. **Grid interpolation** — velocities (`uo`,`vo`) to tracer cell centers
   (Arakawa C→A). See `archive/DataLoader_Input/scripts/messy_convert_to_Arakawa_A_grid.py`.
2. **Variable naming** — ensure state file has `uo,vo,temp,salt,SSH`; fluxes have
   the `dynamic_forcing` names used in the config.
3. **Static file** — confirm `wet` (0/1) and `deptho` present; the loader takes
   `isel(time=0)` so the stray time axis is fine as-is.
4. **Boundary conditions (OBCs)** — already prepared on the model grid upstream;
   just add them to the `dynamic_forcing` variable list in the config. No
   regridding/calendar work needed. (A dedicated OBC input layer is a future
   option — see §2.)
5. **Statistics / scaler** — run `credit preprocess` over the ocean data to fit a
   bridgescaler `standard` (mean/std) `scaler.json` covering every variable, and
   point `scaler_path` in the config's pre/post `bridgescaler_transform` blocks
   at it.
   - ⚠️ **Channel-order caveat:** gen2 derives channel order by name/rank
     (`prognostic < static < dynamic_forcing < diagnostic`, 3D before 2D — see
     `credit/datasets/channel_layout.py`), *not* by the fixed index order the old
     `_reshape_and_concat` used. Stats keyed by name are order-agnostic, so this
     is safe **as long as stats are name-keyed** (they are). Do not reuse any
     old index-keyed clamp/scaler artifacts.

---

## 4. Testing checklist (run once envs exist)

Ordered from cheapest/most-isolated to full integration. Marked whether a GPU or
real data is needed.

> **Already verified during implementation** against the shared env
> `/glade/work/schreck/conda-envs/credit-main-derecho` and the 5-timestep sample
> files in `archive/DataLoader_Input/` (CPU only):
> **T0** imports/registries · **T1** dataset on real data (temp→thetao rename,
> static singleton-time handling, 65-level shapes, targets) · **T4** density
> (rho0 exact + stability monotonicity), clamp (bounds), wet-mask (real static,
> land frac 0.42) · **plus** the integration path
> concat → reconstruct → clamp → wet-mask → density with real tensor shapes.
> These are known-good. **T5–T7 (GPU / full training / rollout) remain to run
> in your env.** Two bugs were found and fixed by this testing: `zl` is not in
> the static file (wet-mask now reads layer depths from `level_source_path`), and
> the example dates must match the sample files' actual timestamps
> (`1999-12-30T12:00` … `2000-01-03T12:00`).

### T0 — Import & registry (no GPU, no data)
```bash
python -c "import credit.datasets.rmom6, credit.postblock"
python -c "from credit.datasets.multi_source import _SOURCE_REGISTRY; assert 'RMOM6' in _SOURCE_REGISTRY"
python -c "from credit.postblock import POSTBLOCK_REGISTRY as R; \
  assert {'ocean_wet_mask','ocean_clamp','ocean_density','mean_std_denormalize'} <= set(R)"
```
Expect: clean imports, assertions pass. (Requires the credit env with torch/xarray.)

### T1 — Dataset unit test (no GPU; needs the example `.nc` files)
Instantiate directly and inspect one sample:
```python
import yaml
from credit.datasets.rmom6 import RegionalMOM6Dataset
conf = yaml.safe_load(open("config/rmom6_regional.yml"))
ds = RegionalMOM6Dataset(conf["data"], return_target=True)
s = ds[(ds.datetimes[0], 0)]
print(sorted(s["input"]))          # keys "rMOM6/prognostic/3d/thetao", etc.
print(s["input"]["rMOM6/prognostic/3d/thetao"].shape)   # (65, 1, 457, 759)
print(s["input"]["rMOM6/prognostic/2d/SSH"].shape)      # (1, 1, 457, 759)
print(s["target"].keys(), s["metadata"])
```
Verify: `thetao` reads `temp` via rename; static loads at `i==0` without a
KeyError on the stray time axis; 3D tensors have 65 levels; NaNs present (land)
— they get filled downstream, not here.

### T2 — Preblock chain (no GPU; needs stats files from §3.5)
```python
from credit.preblock import build_preblocks, apply_preblocks
pre = build_preblocks(conf["preblocks"], phase="per_step")
# wrap sample as a batch of 1 and run:
out = apply_preblocks(pre, {"input": {"rMOM6": {k: v.unsqueeze(0) for k,v in s["input"].items()}}})
print(out["x"].shape)   # (1, C, H, W) — C = concatenated channel count
```
Verify: normalization applied (values ~O(1)); no NaNs after `fill_values`;
concat channel count = sum of levels×vars in canonical order.

### T3 — Postblock chain incl. ocean blocks (no GPU; synthetic tensors OK)
Build a fake `full_data_dict` with a random `y_pred` and the `_channel_map` from
T2's metadata, then:
```python
from credit.postblock import build_postblocks, apply_postblocks
step = build_postblocks(conf["postblocks"], phase="per_step")
roll = build_postblocks(conf["postblocks"], phase="post_rollout")
fdd = apply_postblocks(step, fake_full_data_dict)
fdd = apply_postblocks(roll, fdd)
yp = fdd["y_processed"]["rMOM6"]
# checks:
#  - denorm returns physical ranges (thetao ~[-2,40], so ~[0,45])
#  - clamp: no values outside configured bounds
#  - wet_mask: land cells (deptho NaN/0) are exactly 0
#  - density: "rMOM6/diagnostic/3d/rho" present, ~[1015,1030] kg/m^3
```

### T4 — Unit tests for the physics blocks (no GPU, no data)
Pure-tensor tests, no files:
- **OceanDensity**: feed constant `theta=theta0, salt=s0` → expect `rho == rho0`.
  Feed a warm-over-cold unstable column with `enforce_stability=True` → assert
  `rho` non-decreasing along dim=1.
- **OceanClamp**: tensor with out-of-range values → assert clamped exactly.
- **OceanWetMask**: build a tiny static file with a known `wet`/`deptho` → assert
  masked tensor is 0 exactly where dry, unchanged where wet.
Add these under `tests/` mirroring existing `tests/test_postblock*` style.

### T5 — Model geometry / forward pass (GPU; small)
The wxformer window/stride/padding in `config/rmom6_regional.yml` is a
**placeholder**. Verify the padded 457×759 grid divides the window stack at each
stage, or adjust `pad_lat`/`pad_lon`/`global_window_size`. Run a single forward
pass with `train_batch_size: 1` and confirm output shape matches the target
channel count. Reduce `levels` (upper-ocean subset) if memory is tight.

### T6 — One training step end-to-end (GPU)
```bash
credit train -c config/rmom6_regional.yml   # or the repo's gen2 train entrypoint
```
Watch for: dataloader returns nested dicts; preblocks assemble `x`; forward runs;
postblocks run without shape errors; loss is finite and decreasing over a few
steps. Set `thread_workers: 1`, `prefetch_factor: 1` if startup hangs.

### T7 — Rollout / regression (GPU)
Run a short rollout and confirm clamp + wet-mask keep the autoregressive state
physical (no NaN blow-up, land stays 0). Compare against the archived thesis
outputs (RMSE timeseries PNGs, PSD gifs, `so`/`thetao` animations in
`archive/Thesis_Archive/`) as the qualitative regression reference.

---

## 5. Known limitations / follow-ups

- **Density stability fixer is a placeholder.** `enforce_stability=True` makes the
  *diagnostic density field* monotonic via `cummax` down the column; it does
  **not** back-correct `theta`/`S`. A faithful convective-adjustment fixer that
  mixes T/S mass-weighted by layer thickness (as MOM6 does) is the real future
  work. Keep it `false` until validated (T4/T7).
- **Linear EOS.** `ocean_density` uses a linear Boussinesq potential-density EOS.
  Swap in a TEOS-10 / Roquet-2015 torch polynomial in `_eos_linear` for accuracy;
  keep it differentiable (no `gsw`).
- **Gradient detachment.** See §2 — physical postblocks don't shape training
  gradients by design of `Reconstruct`. Revisit if training-time constraints are
  wanted.
- **Model geometry** (`config/rmom6_regional.yml` `model:`) is unvalidated — T5.
- **Loss masking.** MSE currently includes land points (filled to 0). Consider a
  wet-masked loss so land doesn't dilute the signal (not implemented here).
- **Manual C-to-A regrid, no xgcm dependency anywhere in this pipeline (deliberate, revisit
  later).** `build_rmom6_pointwise_scaler.py` (added for the pointwise xi-normalization
  experiment, see `EXPERIMENT_DESIGN.md`) needed to regrid `stats_pointwise_uo.nc`/
  `stats_pointwise_vo.nc` from the raw archive's native Arakawa-C face grid (`xq`/`yq`) onto
  the tracer-center A-grid (`xh`/`yh`) the dataset actually emits, and did so with a plain
  numpy center-average (`0.5*(face[:-1]+face[1:])`) rather than xgcm, since it runs in
  `miles-credit-casper`, which doesn't have xgcm installed. `preprocess_rmom6.py` originally
  used `xgcm.Grid(...).interp(da, axis, to="center")` for the same regrid on the main
  prognostic/forcing data and required a separate `npl` env for it; it has since been switched
  to the identical manual formula (`center_average()` in that script) so the whole pipeline
  now runs in one env with no xgcm dependency at all. This is a deliberate, temporary choice —
  the manual formula matches xgcm's `interp` for this domain's symmetric grid, but relies on
  that assumption holding (N+1 face points bracketing N tracer cells, no holes/irregular
  masking at the regrid step) rather than xgcm's more general handling. **Future work: bring
  xgcm back** (as an actual `miles-credit` dependency, so no env-switching is needed) if a grid
  arises where that assumption doesn't hold. Verified bit-exact against the prior xgcm output
  for uo/vo/thetao/so/SSH before switching `preprocess_rmom6.py` over (max abs diff `0.0`).
- **`stats_pointwise_uo.nc`/`vo.nc`'s `std` is an approximation, not exact, for the A-grid.**
  `explore_statistics_carib12/compute_stats.py` computes these directly from the raw archive
  (no regrid), so they're natively on uo/vo's C-grid face dims (`xq`/`yq`). `mean` is exact
  either way — time-averaging and center-averaging commute (`mean` is linear) — but
  `build_rmom6_pointwise_scaler.py`'s center-average of the *std* values
  (`0.5*(std[i]+std[i+1])`) is only the true std of the actual A-grid-averaged variable if the
  two adjacent C-grid points are perfectly correlated in time; in general
  `Var(0.5*(A+B)) = 0.25*(Var(A)+Var(B)+2*Cov(A,B))`, and this shortcut implicitly assumes
  `Cov(A,B) ≈ Var(A) ≈ Var(B)`. For grid-scale-resolved currents this is a reasonable but not
  exact assumption — the true A-grid std could be up to ~30% smaller than this approximation
  in the limit of fully-uncorrelated neighbors. **Not fixed here** (would require regridding
  uo/vo in `compute_stats.py`'s `open_3d()` before computing stats and re-running the full
  20-year stats pipeline) — `thetao`/`so`/`SSH` are unaffected (already tracer-grid natively),
  and `stats_levelwise.nc`/`stats_xi.nc` are unaffected too (they reduce over all spatial dims
  before the grid distinction would ever matter). Revisit if pointwise-normalized uo/vo look
  mis-scaled relative to levelwise.
- **Open boundary conditions (OBCs) — naive edge-overwrite in place, dedicated pathway still
  needed.** Until this note, OBCs were not connected to the pipeline at all (§3 already flagged
  this): `scripts/preprocess_obcs.py` produces `obc_{direction}[_sample].zarr`, but nothing read
  them. Added `ocean_obc_nudge` (`credit/postblock/ocean_obc_nudge.py`, registered as
  `ocean_obc_nudge`) as a **naive placeholder**: every `per_step` postblock pass, it literally
  overwrites the single-pixel-wide outer edge ring of `thetao`/`so`/`uo`/`vo`/`SSH` in
  `y_processed` with the prescribed OBC value at the current step's real timestamp (nearest-
  matched against each `obc_{direction}.zarr`'s own `time` coordinate). This required a small
  fix in `credit/trainers/rollout_utils.py`'s `assemble_rollout_batch`: it used to silently drop
  the `"metadata"` key from its returned dict, so `batch_dict["metadata"]["target"][source]
  ["datetime"]` was only ever populated at rollout step 0 (verified: `ConcatToTensor`,
  `credit/preblock/concat.py:93-100`, only populates `"datetime"` when a top-level `"metadata"`
  key is present in the batch it's given) — every step after that silently had no timestamp
  available. Now `assemble_rollout_batch` forwards `curr_batch.get("metadata")` through, same
  as it already did for `"target"`.
  - **This is not a real fix, just a stopgap.** It has no relaxation timescale / smooth
    sponge-layer blend into the interior (real regional-model OBC treatments nudge over a
    boundary *zone*, not one pixel), and — more importantly — it does not give the **model**
    any information about the boundary condition as an *input*; the model still has to predict
    the interior having never seen what's coming in from outside, and the overwrite only
    happens to the postblock-processed output that feeds the *next* step, not the current
    step's forward pass or loss.
  - **Real fix, future work — dedicated OBC input pathway.** Two options sketched during
    design (see the conversation this session, not yet built): (1) scatter each boundary strip
    into a full-domain-shaped field (zero/NaN off the boundary) at preprocessing time so it can
    ride in as an ordinary `dynamic_forcing` variable through the existing `MultiSourceDataset` →
    `ConcatToTensor` pipeline — simplest, no model changes, but wastes most of the channel on
    padding and the model has to learn "only the edge matters" on its own; or (2) a genuinely
    separate OBC source/preblock that keeps the boundary strips in their native 1D shape and
    feeds them into the encoder directly (e.g. a boundary embedding, or an additive correction
    term near the domain edge) — more architecture work, but doesn't waste channels/memory and
    lets the model actually condition on the boundary forcing rather than just being nudged
    after the fact.

---

## 6. Multi-step training (rollout) — how it works here, and questions to raise

**Short answer: yes, multi-step training works with this setup.** Set
`data.forecast_len: N` (N > 1) and the gen2 trainer rolls the model forward N
steps per sample. The regional ocean dataset + preblocks + postblocks all
participate correctly. But there are two behaviors worth understanding (and
worth confirming with the CREDIT team), because they differ from a naive
"backprop through the whole rollout" expectation.

### What actually happens each rollout step (verified in code)
Per training sample, `trainer_gen2.train_one_epoch` runs
`for t in range(1, forecast_len + 1)`:

1. **t = 1:** dataset emits the raw (physical) initial condition → `ic_only`
   preblocks (statics) → `per_step` preblocks (`bridgescaler_transform` →
   `fill_values` → `concat`) build the normalized model input `x`.
2. **Forward:** model produces `y_pred` (flat, normalized, **grad-attached**).
3. **Postblocks (`per_step`):** `reconstruct` splits `y_pred` into
   `y_processed` **and detaches it** (`reconstruct.py:64`); then
   `bridgescaler_transform` (inverse) → `ocean_clamp` → `ocean_wet_mask` bring it
   to physical, bounded, land-masked values.
4. **Loss:** computed on `y_pred` (normalized, attached) vs the normalized target
   — **not** on `y_processed`.
5. **t → t+1:** `assemble_rollout_batch` feeds the previous step's `y_processed`
   (physical) back as the prognostic input, adds the new step's `dynamic_forcing`,
   and the `per_step` preblocks **re-normalize** it for the next forward.

### Behavior #1 — gradients do NOT flow across rollout steps (truncated BPTT)
Because `reconstruct` **detaches** `y_processed`, and the next step's input is
built from `y_processed`, the input to step *t+1* is a constant to autograd. So
each step's loss trains only that step's forward pass — this is
**"pushforward" / truncated-BPTT** training, not full backprop-through-time.
`backprop_on_timestep` selects which steps contribute a loss term.

- *Question for the team:* is truncated/pushforward the intended gen2 recipe for
  multi-step, or is there a supported full-BPTT-through-rollout path? Our old
  thesis training — did it rely on gradients propagating across the rollout?

### Behavior #2 — the physical round-trip must stay consistent every step
The normalizer runs **every** step (`per_step`), so whatever is fed back must be
in the **same units the dataset emits (physical)**. That is exactly why the
postblock chain inverse-transforms `y_processed` back to physical: so the next
step's normalizer sees physical values, not already-normalized ones (which would
double-normalize and corrupt the rollout). bridgescaler `transform` ⇄
`inverse_transform` closes this loop cleanly.

- *Alternative the team may suggest:* keep `y_processed` in normalized space
  (skip the inverse) and set the target-side scaler to `data_types: ["target"]`
  so the already-scaled input isn't re-scaled (bridgescaler supports this). We
  **did not** use that path because `ocean_clamp`/`ocean_density` need physical
  units in `per_step`; if those physics blocks move to `post_rollout` instead,
  the normalized-feedback path becomes viable. Worth asking which they recommend.

### Practical checklist to turn multi-step on
- Set `data.forecast_len` (and `valid_forecast_len`) to N > 1. The gen2 loader's
  `DistributedMultiStepBatchSampler(num_forecast_steps=forecast_len)` yields the
  `(t, i)` sequences; the dataset already returns prognostic+static at `i==0`
  and `dynamic_forcing` every step.
- Ensure enough consecutive timesteps exist per sample (the 5-step example files
  only support very short rollouts).
- `reconstruct` **must** be the first postblock (else `assemble_rollout_batch`
  raises) — it already is in `config/rmom6_regional.yml`.
- Watch memory: N steps hold N forward graphs for the backprop steps.
