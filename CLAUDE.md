# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MILES CREDIT (NSF NCAR) — a platform for training and deploying AI Earth-system emulators (atmosphere, ocean,
regional WRF/LES downscaling). Configurable input data, network architecture, training loop, and pre/post
physics constraints, driven entirely by YAML config files. Runs on NCAR Casper/Derecho (PBS) as well as generic
single/multi-GPU machines.

## Commands

```bash
# Install (editable, into an existing env with heavy deps like torch/xarray preinstalled)
pip install -e . --no-deps

# Dev install with lint/test/docs extras
pip install ".[develop]"

# Lint / format (also runs automatically via pre-commit)
ruff check --fix
ruff format

# Tests (CI runs this; excludes tests/manual which needs multi-GPU/HPC hardware)
pytest
pytest tests/test_postblock.py                       # one file
pytest tests/test_postblock.py::test_some_case -v    # one test
pytest --cov=credit --cov-branch --cov-report=xml    # with coverage, as CI does

# Unified CLI (entry point: `credit`, see credit/cli/)
credit init --grid 1deg -o my_run.yml        # generate a config
credit train -c my_run.yml                   # train
credit rollout -c my_run.yml                 # rollout/inference
credit realtime -c my_run.yml --init-time ... --steps ...
credit submit --cluster derecho -c my_run.yml --gpus 4 --nodes 1   # PBS job (chains jobs via afterok)
credit plot -c my_run.yml --field VAR_2T --denorm
credit ask "why is my loss not decreasing?"  # agent/chat assistant, see credit/cli/_ask.py
```

Legacy (gen1) console scripts also exist (`credit_train_gen1`, `credit_rollout_realtime`, `credit_train_wrf`,
`credit_rollout_les`, etc. — see `[project.scripts]` in `pyproject.toml`); prefer the unified `credit` CLI for
new work.

`tests/manual/` holds multi-GPU/MPI tests that CI cannot run (`norecursedirs` in `pyproject.toml`) — run these
by hand on a GPU node when touching `credit/domain_parallel/` or `credit/parallel/`.

## Architecture

There are two config/pipeline generations living side by side. New work should target **gen2**.

### Gen2 pipeline (current)

Data flows through four config-driven, registry-based stages. Each stage type has its own `_REGISTRY` dict
(`credit/datasets/multi_source.py:_SOURCE_REGISTRY`, `credit/preblock/__init__.py:PREBLOCK_REGISTRY`,
`credit/models/__init__.py:_MODEL_REGISTRY`, `credit/postblock/__init__.py:POSTBLOCK_REGISTRY`) mapping a config
`type` string to a lazily-imported class — this is the extension point for adding a new data source, transform,
model, or physics constraint without touching the pipeline itself.

1. **`MultiSourceDataset`** (`credit/datasets/multi_source.py`) — wraps one or more named sources
   (`config["data"]["source"][<name>]`, each with a `dataset_type` routed through `_SOURCE_REGISTRY`), returns a
   sample dict nested as `{"input": {source: {"<source>/<field_type>/<dim>/<var>": tensor}}}` (field types:
   `prognostic`, `dynamic_forcing`, `static`, `diagnostic`). Most concrete sources (`LocalDataset`, ocean
   `RegionalMOM6Dataset`, etc.) subclass `AbstractBaseDataset`/`BaseDataset`.
2. **Preblocks** (`credit/preblock/`) — per-source-name transforms run before concatenation
   (`ic_only` phase runs once at t=0, e.g. static regrid; `per_step` runs every rollout step, e.g. normalization,
   `fill_values`, then `ConcatToTensor` which flattens the nested dict into the model input tensor `x`). Channel
   order in the concatenated tensor follows a fixed cross-group rank — see `FIELD_TYPE_RANK` in
   `credit/datasets/channel_layout.py` (`prognostic < static < dynamic_forcing < diagnostic`, 3D vars before 2D
   within a group) — **not** config key order. `build_channel_layout()`/`update_x()` in that module derive
   per-group slices for reassembling the next rollout step's input from the model's prediction plus new forcing.
3. **Model** — plain `nn.Module`, selected via `model.type` in config and routed through `_MODEL_REGISTRY`
   (`credit/models/__init__.py`). Register a custom model with `@register_model("name")` and point
   `custom_models:` in the config at the file. WxFormer/CrossFormer is the flagship architecture; see
   `docs/source/Model_Architectures.md`.
4. **Postblocks** (`credit/postblock/`) — run on the model's raw prediction. `per_step` phase runs after every
   forward pass in the rollout loop (must start with `reconstruct`, which splits `y_pred` back into the nested
   `y_processed[source][var_key]` dict and **detaches it from autograd** — physics/clamp/mask postblocks shape
   rollout feedback and saved output, not the training gradient); `post_rollout` runs once after the full
   rollout. Multi-step training is therefore **truncated-BPTT ("pushforward")**, not full backprop through the
   rollout — `backprop_on_timestep` in the trainer selects which steps get a loss term.

`TrainerERA5Gen2` (`credit/trainers/trainer_gen2.py`) drives this loop; `assemble_rollout_batch`
(`credit/trainers/rollout_utils.py`) builds each subsequent step's input from the previous step's
(inverse-transformed, physical-space) `y_processed` plus newly-loaded `dynamic_forcing`.

### Gen1 pipeline (legacy)

Flat (non-nested) data schema, separate trainer classes per model family under `credit/trainers/`
(`trainerERA5gen1.py`, `trainerWRF.py`, `trainerLES.py`, `trainerERA5_ensemble.py`,
`trainerERA5_Diffusion.py`, `trainer_om4_samudra.py`), config validation centered in `credit/parser.py`. Kept for
existing configs/reproducibility (`config/gen_1/`); don't build new features on it.

### Parallelism

`credit/parallel/` (FSDP2, tensor-parallel mesh, DDP-style collectives) and `credit/domain_parallel/` (spatial
domain decomposition with halo exchange for conv layers — `layers.py`, `halo_exchange.py`, `manager.py`) are
composable independently. `get_domain_manager()`/`get_raw_model()` (`credit/parallel/domain.py`) unwrap whichever
combination of FSDP2/DDP/domain-parallel wrapping is active so trainer code doesn't need to know the wrapping
strategy.

### Config layout

`config/gen_2/examples/` has working starting configs (`example-v2026.2.yml` is the fully annotated reference);
`config/gen_1/` is organized by application (climate, diffusion, downscaling, ensemble, ic_opt, physics, etc.) and
kept for reference/reproducibility only. PBS submission settings live under a config's `pbs:` block and are
consumed by `credit submit`. See `config/README.md` for the full directory map.

## Conventions

- Python >= 3.11, ruff line length 120, ruff-format for formatting (both run via pre-commit).
- New data sources / preblocks / postblocks / models are added by registering into the relevant `_REGISTRY` dict
  and lazily importing the implementation module — follow the existing pattern rather than importing eagerly at
  package init (keeps optional heavy deps like `gcsfs`/`herbie`/`s3fs` out of the default import path).
- Stats/scalers in the gen2 pipeline (bridgescaler) are keyed by variable **name**, not positional index — do not
  reintroduce index-keyed artifacts, since channel order is derived by rank (see `channel_layout.py`), not config
  order.
