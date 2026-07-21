"""Shared helpers for pointwise (per-level, per-pixel) normalization.

bridgescaler's ``DStandardScalerTensor`` (used by ``bridgescaler_transform``) only supports
per-channel statistics broadcast uniformly across H/W (see ``reshape_to_channels_first`` in
``bridgescaler.distributed_tensor``) -- it cannot represent a mean/std that also varies
spatially. ``PointwiseScalerTransform`` (preblock in this module's sibling file, postblock in
``credit/postblock/pointwise_scaler.py``) instead loads a torch-saved dict of full ``(nz, H,
W)`` mean/std tensors per variable and applies the transform elementwise.

Stats file format (as produced by ``scripts/build_rmom6_pointwise_scaler.py``), loaded with
``torch.load``::

    {var_key: {"mu": tensor(nz, H, W), "sigma": tensor(nz, H, W)}, ...}

``nz`` is 1 for 2D variables (e.g. SSH). Land/invalid points are expected to already be filled
(mu=0, sigma=1) by the build script -- this module does no NaN handling of its own.

Shared by both blocks -- lives in ``preblock`` and is imported by ``postblock/pointwise_scaler.py``,
mirroring how ``_parse_variable_selection`` is shared for ``bridgescaler_transform``.
"""

from os.path import expandvars

import torch

_METHODS = ("transform", "inverse_transform")


def load_pointwise_stats(stats_path: str) -> dict:
    return torch.load(expandvars(stats_path), map_location="cpu")


def apply_pointwise_scale(x: torch.Tensor, stat_entry: dict, method: str) -> torch.Tensor:
    """Apply (or invert) elementwise standardization against a tensor shaped (B, nz, T, H, W).

    ``stat_entry["mu"]``/``["sigma"]`` are shaped ``(nz, H, W)`` (``nz=1`` for 2D vars) and are
    unsqueezed to ``(1, nz, 1, H, W)`` to broadcast across the batch and time/history dims.
    """
    if method not in _METHODS:
        raise ValueError(f"apply_pointwise_scale: method must be one of {_METHODS}, got {method!r}")
    mu = stat_entry["mu"].to(dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(2)
    sigma = stat_entry["sigma"].to(dtype=x.dtype, device=x.device).unsqueeze(0).unsqueeze(2)
    if method == "transform":
        return (x - mu) / sigma
    return x * sigma + mu


def get_cached_stat_entry(
    cache: dict, stat_entry: dict, var_key: str, dtype: torch.dtype, device: torch.device
) -> dict:
    """Return stat_entry's mu/sigma moved to (dtype, device), cached by (var_key, device, dtype).

    Avoids re-transferring the full (nz, H, W) mu/sigma tensors to the GPU on every forward call
    -- ``apply_pointwise_scale``'s own ``.to()`` call becomes a no-op once the cached entry
    already matches, since ``Tensor.to()`` returns ``self`` when dtype/device already match.
    """
    key = (var_key, device, dtype)
    cached = cache.get(key)
    if cached is None:
        cached = {
            "mu": stat_entry["mu"].to(dtype=dtype, device=device),
            "sigma": stat_entry["sigma"].to(dtype=dtype, device=device),
        }
        cache[key] = cached
    return cached
