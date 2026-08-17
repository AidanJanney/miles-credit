import gc
import logging
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import tqdm

import optuna

from credit.losses.base_losses import is_crps_loss
from credit.parallel.domain import (
    gather_spatial,
    get_domain_manager,
    get_raw_model,
    shard_spatial,
    unpad_shard_interp,
    sync_domain_gradients,
)
from credit.parallel.collectives import all_reduce_avg, clip_grad_norm_, total_grad_norm
from credit.parallel.fsdp2 import fsdp2_is_applied
from credit.postblock import build_postblocks, apply_postblocks
from credit.preblock import build_preblocks, apply_preblocks
from credit.trainers.rollout_utils import assemble_rollout_batch
from credit.scheduler import update_on_batch
from credit.trainers.base_trainer import BaseTrainer
from credit.trainers.utils import accum_log, cycle

logger = logging.getLogger(__name__)


def _physical_space_mae(y_processed: dict, y_raw: dict) -> dict[str, float]:
    """Per-variable MAE between the postblock-denormalized prediction (``y_processed``) and
    the never-normalized target (``y_raw``), in physical units.

    Training loss/metrics are computed in normalized space, which is not comparable across
    configs using different normalization scales (e.g. CREDIT-style xi tendency scaling vs
    plain std -- see EXPERIMENT_DESIGN.md §6). This is a supplementary diagnostic only, logged
    alongside those metrics; it does not affect training.
    """
    out = {}
    for source, pred_vars in y_processed.items():
        raw_vars = y_raw.get(source, {})
        for var_key, pred in pred_vars.items():
            target = raw_vars.get(var_key)
            if target is None or target.shape != pred.shape:
                continue
            diff = (pred - target.to(device=pred.device, dtype=pred.dtype)).abs()
            finite = torch.isfinite(diff)
            if not finite.any():
                continue
            out[var_key] = diff[finite].mean().item()
    return out


class TrainerERA5Gen2(BaseTrainer):
    def __init__(self, model: torch.nn.Module, rank: int, conf: dict):
        """
        Gen 2 trainer for the ERA5 nested data schema.

        Key differences from TrainerERA5Gen1:
          - Uses new nested data schema: conf["data"]["source"]["ERA5"]["variables"]
          - Applies preblocks to assemble batch tensors before the model forward pass
          - forecast_len semantics: 1 = 1 step (Gen 1 used 0 = 1 step)
          - backprop_on_timestep: range(1, forecast_len+1) instead of range(0, forecast_len+2)
          - Validation config read from conf["validation_data"] if present, else conf["data"]
          - Postblocks applied after model forward pass via apply_postblocks(phase="per_step")
          - Multi-step rollout uses assemble_rollout_batch() + apply_preblocks(phase="per_step") each step

        Args:
            model: The (possibly DDP/FSDP-wrapped) model.
            rank: Global rank of this process.
            conf: Full configuration dict.
        """
        super().__init__(model, rank, conf)

        # The config can request fsdp2 while the wrapper skips it (dp_size <= 1).
        # AMP decisions must follow what was actually applied: when FSDP2 is
        # active its MixedPrecisionPolicy replaces autocast; when it was
        # skipped, plain autocast (trainer.amp) is all the mixed precision
        # this run has.
        self._fsdp2_active = fsdp2_is_applied(model)

        # ---- Domain parallel manager (None when not using domain parallel) ----
        self.domain_manager = get_domain_manager(model)
        self._raw_model = get_raw_model(model)
        raw_m = self._raw_model
        if (
            self.domain_manager is not None
            and self.domain_manager.domain_parallel_size > 1
            and getattr(raw_m, "use_padding", False)
        ):
            self._domain_pre_pad = raw_m.padding_opt
            self._domain_image_h = raw_m.image_height
            self._domain_image_w = raw_m.image_width
            raw_m.use_padding = False
            raw_m.use_interp = False
        else:
            self._domain_pre_pad = None
            self._domain_image_h = None
            self._domain_image_w = None

        preblock_cfg = conf.get("preblocks", {})
        self.ic_preblocks = build_preblocks(preblock_cfg, phase="ic_only")
        self.step_preblocks = build_preblocks(preblock_cfg, phase="per_step")

        postblock_cfg = conf.get("postblocks", {})
        self.step_postblocks = build_postblocks(postblock_cfg, phase="per_step")
        self.rollout_postblocks = build_postblocks(postblock_cfg, phase="post_rollout")

        # ---- Data schema extraction (new nested schema) ----
        data_conf = conf["data"]
        source = next(iter(data_conf["source"].values()))
        vars_conf = source["variables"]
        diag = vars_conf.get("diagnostic") or {}
        num_levels = len(source.get("levels") or [])
        self.varnum_diag = (len(diag.get("vars_3D", [])) * num_levels + len(diag.get("vars_2D", []))) if diag else 0

        self.retain_graph = data_conf.get("retain_graph", False)

        # forecast_len: 1 = 1 step (new semantics, unlike v1 where 0 = 1 step)
        self.forecast_len = data_conf["forecast_len"]
        trainer_conf = conf.get("trainer", {})
        bpt = trainer_conf.get("backprop_on_timestep") or data_conf.get("backprop_on_timestep")
        self.backprop_on_timestep = bpt if bpt is not None else list(range(1, self.forecast_len + 1))
        # How many rollout steps contribute a loss term, i.e. how many times backward() runs
        # per training iteration. Used to average the accumulated gradient over those steps.
        self.n_backprop_steps = max(len(self.backprop_on_timestep), 1)

        data_clamp = data_conf.get("data_clamp")
        if data_clamp is None:
            self.flag_clamp = False
            self.clamp_min = None
            self.clamp_max = None
        else:
            self.flag_clamp = True
            self.clamp_min = float(data_clamp[0])
            self.clamp_max = float(data_clamp[1])

        # Validation config: use validation_data block if present, else fall back to data
        data_valid = conf.get("validation_data", data_conf)
        self.valid_history_len = data_valid.get("history_len", data_conf.get("history_len", 1))
        self.valid_forecast_len = data_valid.get("forecast_len", self.forecast_len)

        # If True, log a warning on NaN loss instead of raising TrialPruned.
        self.skip_nan_prune = conf.get("trainer", {}).get("skip_nan_prune", False)

        loss_name = conf.get("loss", {}).get("training_loss")
        if is_crps_loss(loss_name) and self.ensemble_size <= 1:
            raise ValueError(
                f"{loss_name} is an ensemble CRPS loss and requires trainer.ensemble_size > 1; "
                f"got trainer.ensemble_size={self.ensemble_size}."
            )
        self.is_ring_crps = loss_name == "ring-crps"
        self.is_crps_ensemble = is_crps_loss(loss_name) and self.ensemble_size > 1
        self.use_batch_axis_ensemble = self.ensemble_size > 1 and not self.is_ring_crps

        # ---- Loss masking over cells the dataset never defines (land / below bathymetry) ----
        self.mask_missing_targets = conf.get("loss", {}).get("mask_missing_targets", False)
        # Prescribed-boundary cells to drop from the loss, as {"lat": [start, end], "lon": ...}
        # counts -- same [start, end] convention as model.padding_conf's pad_lat/pad_lon. These
        # are cells whose value is imposed rather than predicted (an OBC halo appended to the
        # grid), so scoring the model on them penalizes it for reproducing a boundary condition
        # it is handed. Applies only when mask_missing_targets is on, since it shares that mask.
        self.exclude_border = conf.get("loss", {}).get("exclude_border") or {}
        if self.exclude_border and not self.mask_missing_targets:
            raise ValueError(
                "loss.exclude_border requires loss.mask_missing_targets: True — the border "
                "exclusion is applied to that mask, and without it the loss is an unmasked mean."
            )
        # Score the metrics over the same cells as the loss. Off by default: turning it on
        # changes every logged acc/rmse/mse/mae, so runs before and after are not comparable,
        # and every other CREDIT config keeps its historical numbers. Worth turning on for a
        # domain with a large undefined fraction -- on the rMOM6 grid land is 50.75% of cells
        # and is predicted perfectly by construction, which makes the unmasked mae read ~2.03x
        # better than the ocean-only error (rmse ~1.44x; acc is barely affected).
        self.mask_metrics = conf.get("loss", {}).get("mask_metrics", False)
        if self.mask_metrics and not self.mask_missing_targets:
            raise ValueError(
                "loss.mask_metrics requires loss.mask_missing_targets: True — the metrics reuse "
                "the loss mask, and there is no mask to reuse without it."
            )
        self._loss_mask = None
        self._loss_mask_key = None
        # The model's grid, which a preblock may have enlarged past the dataset's (the
        # ocean_obc_halo preblock appends a prescribed boundary ring to both x and y). The loss
        # mask is derived from the *raw* target, so it comes out at the dataset's grid and has
        # to be padded up to this one before it can multiply an elementwise loss.
        self._model_grid = (conf.get("model", {}).get("image_height"), conf.get("model", {}).get("image_width"))
        if self.mask_missing_targets and not any(
            "mask" in type(block).__name__.lower() for block in self.step_postblocks.values()
        ):
            logger.warning(
                "loss.mask_missing_targets is on but no masking postblock is configured in "
                "postblocks.per_step. Masked cells receive no gradient, so the model's output "
                "there is unconstrained and will be fed back into the next rollout step as-is."
            )

    # ------------------------------------------------------------------
    # Domain-parallel forward helpers (shared by train and validate)
    # ------------------------------------------------------------------

    def _sharded_forward(self, full_data_dict, amp_enabled):
        """Model forward with domain-parallel pad/shard handling.

        Pads + shards the input over the domain group, runs the model with its
        internal padding suppressed (the trainer pre-pads the full grid), unpads
        the prediction back to the shard's target shape, flattens 5D output to
        4D, and shards y to match y_pred's spatial shard. A no-op passthrough
        (plus the forward) when domain parallelism is off.
        """
        if self._domain_pre_pad is not None:
            full_data_dict["x"] = self._domain_pre_pad.pad(full_data_dict["x"])
        full_data_dict["x"] = shard_spatial(full_data_dict["x"], self.domain_manager)

        if self._domain_pre_pad is not None:
            self._raw_model._skip_internal_padding = True
        try:
            with torch.autocast(device_type="cuda", enabled=amp_enabled):
                full_data_dict["y_pred"] = self.model(full_data_dict["x"])
        finally:
            if self._domain_pre_pad is not None:
                self._raw_model._skip_internal_padding = False
        if self._domain_pre_pad is not None:
            full_data_dict["y_pred"] = unpad_shard_interp(
                full_data_dict["y_pred"],
                self._domain_pre_pad,
                self.domain_manager,
                self._domain_image_h,
                self._domain_image_w,
            )
        if full_data_dict["y_pred"].dim() == 5:
            full_data_dict["y_pred"] = full_data_dict["y_pred"].flatten(1, 2)

        # domain parallel: shard y to match y_pred's spatial shard
        if "y" in full_data_dict and full_data_dict["y"] is not None:
            _y = full_data_dict["y"]
            if _y.dim() == 5:
                _y = _y.flatten(1, 2)
            full_data_dict["y"] = shard_spatial(_y, self.domain_manager)

    # ------------------------------------------------------------------
    # Loss masking (shared by train and validate)
    # ------------------------------------------------------------------

    def _pad_mask_to_model_grid(self, mask):
        """Zero-pad a raw-target-derived mask up to the model grid, using ``exclude_border``.

        A preblock can hand the model a larger grid than the dataset provides: ``ocean_obc_halo``
        appends a one-cell ring of prescribed open-boundary values to both ``x`` and ``y``. The
        mask is built from ``y_raw`` — deliberately, so the dataset's own missing-data convention
        stays the single source of truth — and therefore comes out at the dataset's grid, one
        cell short of ``y`` in each padded direction.

        ``loss.exclude_border`` already names those cells, in the same ``[start, end]``
        convention the preblock's ``pad_lat``/``pad_lon`` use, and their mask value is zero
        either way, so it doubles as the padding spec here. Getting this wrong is not a
        cosmetic mismatch: without the pad the shape check below rejects the mask outright,
        and padding at the wrong end would silently score the model on a prescribed ring while
        dropping a real interior row.
        """
        H_model, W_model = self._model_grid
        if H_model is None or W_model is None:
            return mask
        dH, dW = H_model - mask.shape[-2], W_model - mask.shape[-1]
        if dH == 0 and dW == 0:
            return mask
        if dH < 0 or dW < 0:
            raise ValueError(
                f"Loss mask grid {tuple(mask.shape[-2:])} is larger than model.image_height/width "
                f"({H_model}, {W_model}); the raw target cannot exceed the model grid."
            )
        lat0, lat1 = self.exclude_border.get("lat", (0, 0))
        lon0, lon1 = self.exclude_border.get("lon", (0, 0))
        if (lat0 + lat1, lon0 + lon1) != (dH, dW):
            raise ValueError(
                f"The model grid ({H_model}, {W_model}) is larger than the raw target grid "
                f"{tuple(mask.shape[-2:])} by ({dH}, {dW}), but loss.exclude_border "
                f"{self.exclude_border or '{}'} accounts for ({lat0 + lat1}, {lon0 + lon1}). "
                "Set loss.exclude_border to the preblock's pad_lat/pad_lon so the loss mask "
                "knows which cells were appended."
            )
        # F.pad takes the last dim first; value 0 marks the appended cells invalid.
        return torch.nn.functional.pad(mask, (lon0, lon1, lat0, lat1), value=0.0)

    def _metrics_mask_kwarg(self, full_data_dict):
        """``{"mask": ...}`` when metrics are masked, otherwise ``{}``.

        Returned as kwargs rather than a plain value so the metrics call signature is untouched
        when the flag is off: not every metrics object in the codebase (or in a test) accepts a
        ``mask`` argument, and a caller that never asked for masking should not have to. The
        mask itself is cached by ``_get_loss_mask``, so this costs a dict lookup per step.
        """
        if not self.mask_metrics:
            return {}
        return {"mask": self._get_loss_mask(full_data_dict)}

    def _get_loss_mask(self, full_data_dict):
        """Cached ``(1, C, H, W)`` float mask of target cells the dataset actually defines.

        Derived from the **raw** target (``y_raw``, i.e. ``batch["target"]`` before the
        ``fill_values`` preblock replaces NaN with 0), so the dataset's own missing-data
        convention is the single source of truth — the same principle the wet-mask fix rests
        on. Channel positions come from ``metadata["target"]["_channel_map"]``, which
        ``ConcatToTensor`` builds alongside ``y`` itself, so the mask cannot drift out of sync
        with concat order the way a separately-configured geometry file could.

        Built once and cached: for an ocean/land grid the missing cells are fixed geometry, so
        rebuilding per batch would be pure overhead. The cache key includes the sharded shape
        and device, so a change in either rebuilds.

        A variable present in the channel map but absent from ``y_raw`` is left fully unmasked
        (all ones) rather than dropped, so an unexpected schema gap cannot silently delete a
        variable from the loss.
        """
        y = full_data_dict["y"]
        key = (tuple(y.shape[1:]), y.device, y.dtype)
        if self._loss_mask_key == key:
            return self._loss_mask

        channel_map = full_data_dict["metadata"]["target"]["_channel_map"]
        raw_vars = {
            var_key: tensor
            for source_vars in full_data_dict["y_raw"].values()
            for var_key, tensor in source_vars.items()
        }

        n_channels = max(entry["slice"].stop for entry in channel_map.values())
        mask = None
        missing = []
        for var_key, entry in channel_map.items():
            tensor = raw_vars.get(var_key)
            if tensor is None:
                missing.append(var_key)
                continue
            # (B, n_levels, T, H, W) -> (1, n_levels * T, H, W), matching the flatten(1, 2)
            # that _sharded_forward applies to y. A cell counts as valid only where it is
            # finite in every sample of the batch.
            var_mask = torch.isfinite(tensor).all(dim=0, keepdim=True).flatten(1, 2)
            if mask is None:
                mask = torch.ones(
                    (1, n_channels, var_mask.shape[-2], var_mask.shape[-1]),
                    dtype=y.dtype,
                    device=var_mask.device,
                )
            mask[:, entry["slice"]] = var_mask.to(mask.dtype)

        if mask is None:
            raise ValueError(
                "loss.mask_missing_targets is on but no target variable in the channel map was "
                f"found in y_raw (channel map keys: {sorted(channel_map)})."
            )
        if missing:
            logger.warning("Loss mask: no raw target for %s; those channels are left unmasked.", sorted(missing))

        mask = self._pad_mask_to_model_grid(mask)

        # Drop prescribed-boundary rows/columns before sharding, while the mask is still on the
        # full grid — after shard_spatial each rank holds only a slice of H and "the last row"
        # is no longer well defined.
        if self.exclude_border:
            lat0, lat1 = self.exclude_border.get("lat", (0, 0))
            lon0, lon1 = self.exclude_border.get("lon", (0, 0))
            H, W = mask.shape[-2], mask.shape[-1]
            if lat0 + lat1 >= H or lon0 + lon1 >= W:
                raise ValueError(f"loss.exclude_border {self.exclude_border} removes the entire {H}x{W} grid.")
            if lat0:
                mask[..., :lat0, :] = 0
            if lat1:
                mask[..., H - lat1 :, :] = 0
            if lon0:
                mask[..., :, :lon0] = 0
            if lon1:
                mask[..., :, W - lon1 :] = 0
            logger.info("Loss mask: excluded border lat=%s lon=%s from a %dx%d grid.", (lat0, lat1), (lon0, lon1), H, W)

        # Domain parallel shards y along H in _sharded_forward; the mask must follow.
        mask = shard_spatial(mask.to(y.device), self.domain_manager)
        if mask.shape[1:] != y.shape[1:]:
            raise ValueError(f"Loss mask shape {tuple(mask.shape)} is incompatible with y {tuple(y.shape)}.")

        # One-time guard: the mask zeroes the loss at invalid cells, but 0 * NaN is NaN in both
        # the forward and the backward pass, so masking cannot rescue a y that still holds NaN.
        # Enabling this flag without a fill_values preblock would otherwise produce a silently
        # NaN loss; fail loudly at the source instead.
        if not torch.isfinite(y).all():
            raise ValueError(
                "loss.mask_missing_targets requires a finite target tensor, but y contains "
                "non-finite values. Add a fill_values preblock (before concat) so missing cells "
                "are filled in normalized space."
            )

        valid_fraction = mask.mean().item()
        logger.info(
            "Loss mask built from raw targets: %.1f%% of %d channels x %d x %d cells are valid.",
            100.0 * valid_fraction,
            mask.shape[1],
            mask.shape[2],
            mask.shape[3],
        )
        if valid_fraction == 0.0:
            raise ValueError("Loss mask is empty — every target cell is non-finite.")

        self._loss_mask = mask
        self._loss_mask_key = key
        return mask

    def _reduce_loss(self, elementwise_loss, full_data_dict):
        """Reduce an elementwise loss to a scalar, optionally ignoring undefined target cells.

        Without ``loss.mask_missing_targets`` this is a plain mean, unchanged. With it, the
        mean is taken over valid cells only: for the regional MOM6 grid roughly half of every
        3D field is land or below-bathymetry geometry that ``fill_values`` sets to a constant
        0 in both ``y`` and (after training) ``y_pred``, so an unmasked mean spends half its
        budget fitting a constant. The mask broadcasts over the batch axis, which also covers
        the ensemble case where ``y_pred`` has ``B * ensemble_size`` rows.

        Note this reduces the loss only — ``metrics`` stays unmasked, so metric columns in
        ``training_log.csv`` keep their previous meaning while ``train_loss``/``valid_loss``
        change scale (roughly doubling for rMOM6, where about half the cells are masked).
        """
        if not self.mask_missing_targets:
            return elementwise_loss.mean()
        if elementwise_loss.dim() == 0:
            raise ValueError(
                "loss.mask_missing_targets requires an elementwise loss, but the criterion "
                "returned a scalar. Losses that reduce internally (VariableTotalLoss2D via "
                "use_latitude_weights / use_variable_weights, ring-crps) cannot be masked here."
            )
        mask = self._get_loss_mask(full_data_dict).to(elementwise_loss.dtype)
        return (elementwise_loss * mask).sum() / mask.expand_as(elementwise_loss).sum().clamp_min(1.0)

    def _gather_for_next_step(self, full_data_dict):
        """Gather domain-sharded y_processed back to full height between rollout steps.

        assemble_rollout_batch concats the previous step's prognostics with
        full-height forcings/statics, so every domain rank needs the full-grid
        prediction; shard_spatial re-shards the assembled input at the next
        forward. No-op without domain parallelism. Gradient-free by design:
        Reconstruct detaches y_processed.
        """
        if self.domain_manager is None or self.domain_manager.domain_parallel_size <= 1:
            return
        y_processed = full_data_dict.get("y_processed")
        if not isinstance(y_processed, dict):
            return
        for source_vars in y_processed.values():
            for var_key, tensor in source_vars.items():
                source_vars[var_key] = gather_spatial(tensor, self.domain_manager)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_one_epoch(self, epoch, trainloader, optimizer, criterion, scaler, scheduler, metrics):
        """
        Train for one epoch.

        The inner loop iterates over forecast_len autoregressive steps. For each step:
          1. Pull the next batch from the dataloader (raw, unnormalized).
          2. At t=1: IC-only preblocks produce ic_preprocessed (regridded statics);
             rollout preblocks produce the final normalized input x.
             At t>1: assemble rollout batch from corrected_pred (prognostic),
             ic_preprocessed (statics), and curr_batch (dynamic forcing);
             rollout preblocks normalize and concat.
          3. Forward pass → y_pred_flat (flat, normalized).
          4. Apply postblocks: Reconstruct → inverse scaler → physics fixers.
             After this, full_data_dict["y_processed"] is a nested dict split by Reconstruct.
          5. Compute loss on y_pred_flat vs the normalized target from preblocks.

        Args:
            epoch: Current epoch number.
            trainloader: DataLoader for training.
            optimizer, criterion, scaler, scheduler, metrics: Standard training objects.

        Returns:
            dict: Training metrics for the epoch.
        """
        if self.ensemble_size > 1:
            logger.info(f"ensemble training with ensemble_size {self.ensemble_size}")

        if self.use_scheduler and self.scheduler_type == "lambda":
            scheduler.step()

        from torch.utils.data import IterableDataset

        batches_per_epoch = self.batches_per_epoch
        if not isinstance(trainloader.dataset, IterableDataset):
            if hasattr(trainloader.dataset, "batches_per_epoch"):
                dataset_batches = trainloader.dataset.batches_per_epoch()
            elif hasattr(trainloader.sampler, "batches_per_epoch"):
                dataset_batches = trainloader.sampler.batches_per_epoch()
            else:
                dataset_batches = len(trainloader)
            batches_per_epoch = (
                self.batches_per_epoch if 0 < self.batches_per_epoch < dataset_batches else dataset_batches
            )

        grad_accum_every = self.conf.get("trainer", {}).get("grad_accum_every", 1)

        # Reseed the shared shuffle permutation each epoch. Without this the
        # sampler (now seeded identically on all ranks — required for disjoint
        # sharding) would yield the same order every epoch.
        _sampler = getattr(trainloader, "batch_sampler", None) or getattr(trainloader, "sampler", None)
        if hasattr(_sampler, "set_epoch"):
            _sampler.set_epoch(epoch)

        batch_group_generator = tqdm.tqdm(
            range(batches_per_epoch),
            total=batches_per_epoch,
            leave=True,
            disable=not any(h.level <= logging.INFO for h in logging.getLogger().handlers),
        )
        self.model.train()

        dl = cycle(trainloader)
        results_dict = defaultdict(list)

        for steps in range(batches_per_epoch):
            logs = {}
            loss = 0
            full_data_dict = {}
            # One metrics dict per rollout step, averaged after the loop. With
            # forecast_len 1 that average is the single step, i.e. identical to the
            # previous final-step-only behavior.
            step_metrics = []

            # Suppress gradient sync on non-boundary micro-steps during
            # gradient accumulation — otherwise DDP all-reduces / FSDP2
            # reduce-scatters on every backward, multiplying comms by
            # grad_accum_every. Both wrappers expose a per-iteration flag
            # (what no_sync() toggles), checked at the next forward/backward.
            is_accum_boundary = (steps + 1) % grad_accum_every == 0 or steps == batches_per_epoch - 1
            if grad_accum_every > 1:
                if self.mode == "fsdp2" and hasattr(self.model, "set_requires_gradient_sync"):
                    self.model.set_requires_gradient_sync(is_accum_boundary)
                elif isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                    self.model.require_backward_grad_sync = is_accum_boundary

            for t in range(1, self.forecast_len + 1):
                batch = next(dl)

                if t == 1:
                    full_data_dict["ic_raw"] = batch["input"]
                    full_data_dict["x_raw"] = batch["input"]
                    full_data_dict["y_raw"] = batch["target"]
                    full_data_dict["ic_preprocessed"] = apply_preblocks(self.ic_preblocks, batch, device=self.device)
                    full_data_dict.update(
                        apply_preblocks(self.step_preblocks, full_data_dict["ic_preprocessed"], device=self.device)
                    )
                else:
                    full_data_dict["x_raw"] = batch["input"]
                    full_data_dict["y_raw"] = batch["target"]
                    full_data_dict.update(
                        apply_preblocks(
                            self.step_preblocks, assemble_rollout_batch(full_data_dict, batch), device=self.device
                        )
                    )

                if self.use_batch_axis_ensemble:
                    full_data_dict["x"] = torch.repeat_interleave(full_data_dict["x"], self.ensemble_size, 0)

                if self.flag_clamp:
                    full_data_dict["x"] = torch.clamp(full_data_dict["x"], min=self.clamp_min, max=self.clamp_max)

                # FSDP2's MixedPrecisionPolicy replaces manual autocast (and
                # conflicts with SpectralNorm power-iteration buffers); when
                # FSDP2 was skipped (dp_size <= 1), plain autocast applies.
                _amp = self.amp and not self._fsdp2_active
                self._sharded_forward(full_data_dict, _amp)

                full_data_dict = apply_postblocks(self.step_postblocks, full_data_dict)
                if t < self.forecast_len:
                    self._gather_for_next_step(full_data_dict)

                if t in self.backprop_on_timestep:
                    if self.flag_clamp:
                        full_data_dict["y"] = torch.clamp(
                            full_data_dict["y"].float(), min=self.clamp_min, max=self.clamp_max
                        )
                    with torch.autocast(device_type="cuda", enabled=_amp):
                        loss = self._reduce_loss(
                            criterion(
                                full_data_dict["y"].float().to(full_data_dict["y_pred"].dtype),
                                full_data_dict["y_pred"],
                            ),
                            full_data_dict,
                        )
                    # n_loss counts the steps that contributed, so the sum below can be
                    # turned back into a per-step mean. Without it train_loss is a sum
                    # over forecast_len while valid_loss is a mean, and the two are not
                    # on the same scale for multistep runs.
                    accum_log(logs, {"loss": loss.item(), "n_loss": 1.0})
                    if self.is_crps_ensemble:
                        # Ensemble spread proxy: std of member errors.
                        target = full_data_dict["y"]
                        if full_data_dict["y_pred"].shape[0] == target.shape[0] * self.ensemble_size:
                            target = torch.repeat_interleave(target, self.ensemble_size, 0)
                        accum_log(logs, {"std": (full_data_dict["y_pred"] - target).detach().std().item()})
                    # Divide by the number of contributing rollout steps as well as by
                    # grad_accum_every. backward() runs once per step and accumulates into the
                    # same .grad, so without this the optimizer sees the SUM over the rollout
                    # while train_loss (logs["loss"] / logs["n_loss"], below) reports the MEAN --
                    # a forecast_len-3 run then takes ~3x the step a forecast_len-1 run takes at
                    # the same trainer.learning_rate, which makes the single-step vs multi-step
                    # arms of an experiment grid incomparable at a shared LR. grad_max_norm does
                    # not absorb it either: "dynamic" clips to the gradient's own norm, i.e. not
                    # at all.
                    scaler.scale(loss / (grad_accum_every * self.n_backprop_steps)).backward(
                        retain_graph=self.retain_graph
                    )

                if full_data_dict.get("y_pred") is not None and full_data_dict.get("y") is not None:
                    step_metrics.append(
                        metrics(
                            full_data_dict["y_pred"],
                            full_data_dict["y"],
                            **self._metrics_mask_kwarg(full_data_dict),
                        )
                    )
                # No barrier here: NCCL collectives (grad sync, halo exchange)
                # already order ranks; a per-timestep barrier only adds latency.

            full_data_dict = apply_postblocks(self.rollout_postblocks, full_data_dict)

            # optimizer step at accumulation boundary
            if is_accum_boundary:
                sync_domain_gradients(self.model, self.domain_manager)
                _tp_group = getattr(self._raw_model, "_tp_group", None)
                if _tp_group is not None:
                    from credit.parallel.tensor_parallel import sync_replicated_gradients

                    sync_replicated_gradients(self.model, _tp_group)
                scaler.unscale_(optimizer)
                if self.grad_max_norm == "dynamic":
                    # Global L2 norm: sum SQUARED norms across ranks, then sqrt.
                    # (Summing the norms themselves and sqrt-ing mixes units.)
                    # DTensor grads (FSDP2 / native TP) go through the
                    # mesh-grouped total_grad_norm, whose full_tensor()
                    # reduction is already global — no extra all_reduce.
                    # Plain grads keep the local sq-sum + SUM all_reduce; that
                    # still over-counts replicated grads when tp/domain ranks
                    # hold copies; acceptable for a clip threshold.
                    from torch.distributed.tensor import DTensor

                    plain, sharded = [], []
                    for p in self.model.parameters():
                        if p.grad is not None:
                            (sharded if isinstance(p.grad, DTensor) else plain).append(p.grad.detach())
                    sq_terms = []
                    if plain:
                        local_sq = torch.stack([g.norm(2) for g in plain]).square().sum()
                        if self.distributed:
                            dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
                        sq_terms.append(local_sq)
                    if sharded:
                        sq_terms.append(total_grad_norm(sharded, 2.0).square())
                    global_norm = torch.stack(sq_terms).sum().sqrt()
                    clip_grad_norm_(self.model.parameters(), max_norm=global_norm)
                elif self.grad_max_norm > 0.0:
                    clip_grad_norm_(self.model.parameters(), max_norm=self.grad_max_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if self.ema is not None:
                    self.ema.update(self.model)

            if step_metrics:
                for name in step_metrics[0]:
                    mean_over_steps = sum(m[name] for m in step_metrics) / len(step_metrics)
                    value = torch.Tensor([mean_over_steps]).to(self.device, non_blocking=True)
                    if self.distributed:
                        all_reduce_avg(value)
                    results_dict[f"train_{name}"].append(value[0].item())

            if full_data_dict.get("y_processed") is not None and full_data_dict.get("y_raw") is not None:
                for var_key, val in _physical_space_mae(full_data_dict["y_processed"], full_data_dict["y_raw"]).items():
                    value = torch.Tensor([val]).to(self.device, non_blocking=True)
                    if self.distributed:
                        all_reduce_avg(value)
                    results_dict[f"train_mae_phys_{var_key.rsplit('/', 1)[-1]}"].append(value[0].item())

            batch_loss = torch.Tensor([logs.get("loss", 0.0) / max(logs.get("n_loss", 1.0), 1.0)]).to(self.device)
            if self.distributed:
                all_reduce_avg(batch_loss)
            results_dict["train_loss"].append(batch_loss[0].item())
            results_dict["train_forecast_len"].append(self.forecast_len)

            if self.is_crps_ensemble and "std" in logs:
                batch_std = torch.Tensor([logs["std"]]).to(self.device)
                if self.distributed:
                    dist.all_reduce(batch_std, dist.ReduceOp.AVG, async_op=False)
                results_dict["train_std"].append(batch_std[0].item())

            if not np.isfinite(np.mean(results_dict["train_loss"])):
                print(results_dict["train_loss"])
                if self.skip_nan_prune:
                    logger.warning("NaN/Inf loss detected but skip_nan_prune=True; continuing.")
                else:
                    raise optuna.TrialPruned()

            self._log_batch_progress(epoch, results_dict, optimizer, batch_group_generator, phase="train")

            if self.use_scheduler and self.scheduler_type in update_on_batch:
                scheduler.step()

        batch_group_generator.close()
        torch.cuda.empty_cache()
        gc.collect()

        return results_dict

    # ------------------------------------------------------------------
    # Validation loop
    # ------------------------------------------------------------------

    def validate(self, epoch, valid_loader, criterion, metrics):
        """
        Validate for one epoch.

        Runs self.valid_forecast_len autoregressive steps per sample.
        Loss and metrics are computed at every step and averaged over the rollout.

        Args:
            epoch: Current epoch number.
            valid_loader: DataLoader for validation.
            criterion, metrics: Loss and metric callables.

        Returns:
            dict: Validation metrics for the epoch.
        """
        self.model.eval()

        from torch.utils.data import IterableDataset

        valid_batches_per_epoch = self.valid_batches_per_epoch
        if not isinstance(valid_loader.dataset, IterableDataset):
            if hasattr(valid_loader.dataset, "batches_per_epoch"):
                dataset_batches = valid_loader.dataset.batches_per_epoch()
            elif hasattr(valid_loader.sampler, "batches_per_epoch"):
                dataset_batches = valid_loader.sampler.batches_per_epoch()
            else:
                dataset_batches = len(valid_loader)
            valid_batches_per_epoch = (
                self.valid_batches_per_epoch if 0 < self.valid_batches_per_epoch < dataset_batches else dataset_batches
            )

        results_dict = defaultdict(list)
        batch_group_generator = tqdm.tqdm(
            range(valid_batches_per_epoch),
            total=valid_batches_per_epoch,
            leave=True,
            disable=not any(h.level <= logging.INFO for h in logging.getLogger().handlers),
        )

        dl = cycle(valid_loader)
        with torch.no_grad():
            for steps in range(valid_batches_per_epoch):
                y_pred_flat = None
                y = None
                loss = 0

                full_data_dict = {}
                step_losses, step_metrics, step_phys = [], [], []

                for t in range(1, self.valid_forecast_len + 1):
                    batch = next(dl)

                    if t == 1:
                        full_data_dict["ic_raw"] = batch["input"]
                        full_data_dict["x_raw"] = batch["input"]
                        full_data_dict["y_raw"] = batch["target"]
                        full_data_dict["ic_preprocessed"] = apply_preblocks(
                            self.ic_preblocks, batch, device=self.device
                        )
                        full_data_dict.update(
                            apply_preblocks(self.step_preblocks, full_data_dict["ic_preprocessed"], device=self.device)
                        )
                    else:
                        full_data_dict["x_raw"] = batch["input"]
                        full_data_dict["y_raw"] = batch["target"]
                        full_data_dict.update(
                            apply_preblocks(
                                self.step_preblocks, assemble_rollout_batch(full_data_dict, batch), device=self.device
                            )
                        )

                    if self.use_batch_axis_ensemble:
                        full_data_dict["x"] = torch.repeat_interleave(full_data_dict["x"], self.ensemble_size, 0)

                    if self.flag_clamp:
                        full_data_dict["x"] = torch.clamp(full_data_dict["x"], min=self.clamp_min, max=self.clamp_max)

                    # Validation runs full precision (no autocast), as before.
                    self._sharded_forward(full_data_dict, False)

                    full_data_dict = apply_postblocks(self.step_postblocks, full_data_dict)
                    if t < self.valid_forecast_len:
                        self._gather_for_next_step(full_data_dict)

                    # Evaluated at every step and averaged after the loop, matching the
                    # train loop. Previously this was gated on t == valid_forecast_len,
                    # so valid_acc reported day-N skill alone while train_loss summed
                    # over all N steps -- the two were never comparable for multistep.
                    if self.flag_clamp:
                        full_data_dict["y"] = torch.clamp(
                            full_data_dict["y"].float(), min=self.clamp_min, max=self.clamp_max
                        )
                    step_losses.append(
                        self._reduce_loss(
                            criterion(
                                full_data_dict["y"].float().to(full_data_dict["y_pred"].dtype),
                                full_data_dict["y_pred"],
                            ),
                            full_data_dict,
                        ).item()
                    )
                    step_metrics.append(
                        metrics(
                            full_data_dict["y_pred"].float(),
                            full_data_dict["y"].float(),
                            **self._metrics_mask_kwarg(full_data_dict),
                        )
                    )
                    if full_data_dict.get("y_processed") is not None and full_data_dict.get("y_raw") is not None:
                        step_phys.append(_physical_space_mae(full_data_dict["y_processed"], full_data_dict["y_raw"]))

                full_data_dict = apply_postblocks(self.rollout_postblocks, full_data_dict)

                for name in step_metrics[0]:
                    value = torch.Tensor([sum(m[name] for m in step_metrics) / len(step_metrics)]).to(
                        self.device, non_blocking=True
                    )
                    if self.distributed:
                        all_reduce_avg(value)
                    results_dict[f"valid_{name}"].append(value[0].item())

                if step_phys:
                    for var_key in step_phys[0]:
                        value = torch.Tensor([sum(p[var_key] for p in step_phys) / len(step_phys)]).to(
                            self.device, non_blocking=True
                        )
                        if self.distributed:
                            all_reduce_avg(value)
                        results_dict[f"valid_mae_phys_{var_key.rsplit('/', 1)[-1]}"].append(value[0].item())

                loss = sum(step_losses) / len(step_losses)
                batch_loss = torch.Tensor([loss]).to(self.device)
                # Average validation loss across ranks (the train loop already
                # does this). Without it each rank tracks a different local
                # valid_loss, and fit() makes early-stopping / best-checkpoint
                # decisions from divergent per-rank histories — ranks can then
                # break out of training at different epochs and hang in the
                # next collective.
                if self.distributed:
                    all_reduce_avg(batch_loss)

                results_dict["valid_loss"].append(batch_loss[0].item())
                results_dict["valid_forecast_len"].append(self.valid_forecast_len)

                self._log_batch_progress(epoch, results_dict, optimizer=None, pbar=batch_group_generator, phase="valid")

        batch_group_generator.close()

        if self.distributed:
            torch.distributed.barrier()

        torch.cuda.empty_cache()
        gc.collect()

        return results_dict


Trainer = TrainerERA5Gen2  # canonical alias, matches other trainer modules
