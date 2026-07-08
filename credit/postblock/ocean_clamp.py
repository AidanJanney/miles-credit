"""
ocean_clamp.py
--------------
OceanClamp: gen2 postblock that clamps reconstructed ocean variables to
physically plausible bounds (e.g. salinity >= 0, temperature within
[-2, 40] degC).

This is the gen2 successor to the ``get_transforms_clamp`` logic in the old
``rMOM6_dataset.py``.  The old code computed clamp tensors in *normalized*
space (subtracting mean / dividing std per variable).  Here we clamp in
*physical* units instead, which is simpler and more interpretable — so this
block must run **after** the inverse-scaler postblock, when ``y_processed``
holds physical values.

It operates on the reconstructed output dict::

    batch_dict["y_processed"][source][var_key] -> tensor (B, n_levels, n_time, H, W)

Bounds are given per variable key.  A missing bound (``null``) means unbounded
on that side.  Because ``Reconstruct`` detaches ``y_processed``, clamping
shapes the rollout feed-back and written outputs, not the single-step training
gradient (see OCEAN_MIGRATION_NOTES.md).

Example config::

    postblocks:
      per_step:
        reconstruct:   { type: reconstruct }
        inverse_scale: { type: bridgescaler_transform, args: {...} }
        clamp:
          type: ocean_clamp
          args:
            bounds:
              "rMOM6/prognostic/3d/so":     [0.0,  45.0]    # salinity, PSU
              "rMOM6/prognostic/3d/thetao": [-2.0, 40.0]    # potential temp, degC
              "rMOM6/prognostic/2d/SSH":    [-5.0, 5.0]     # sea surface height, m
              "rMOM6/prognostic/3d/uo":     [-5.0, 5.0]     # velocity, m/s
              "rMOM6/prognostic/3d/vo":     [-5.0, 5.0]
"""

from __future__ import annotations

import torch

from credit.postblock.base import BasePostblock


class OceanClamp(BasePostblock):
    """Clamp reconstructed ocean variables to per-variable physical bounds.

    Args:
        bounds: Mapping of ``var_key -> [min, max]``.  Either bound may be
            ``null``/``None`` to leave that side unbounded.
        key: Which ``batch_dict`` entry holds the reconstructed dict
            (default ``"y_processed"``).
    """

    def __init__(self, bounds: dict[str, list], key: str = "y_processed"):
        super().__init__()
        self.key = key
        # Normalise to {var_key: (min_or_None, max_or_None)}
        self.bounds: dict[str, tuple] = {}
        for var_key, pair in (bounds or {}).items():
            lo, hi = (pair + [None, None])[:2] if isinstance(pair, list) else (None, None)
            self.bounds[var_key] = (lo, hi)

    def forward(self, batch_dict: dict) -> dict:
        processed = batch_dict.get(self.key)
        if not isinstance(processed, dict):
            return batch_dict

        for source_vars in processed.values():
            for var_key, (lo, hi) in self.bounds.items():
                if var_key not in source_vars:
                    continue
                source_vars[var_key] = torch.clamp(source_vars[var_key], min=lo, max=hi)

        return batch_dict
