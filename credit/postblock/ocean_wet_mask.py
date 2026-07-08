"""
ocean_wet_mask.py
-----------------
OceanWetMask: gen2 postblock that zeroes land points in ocean predictions.

Unlike ``wet_mask_samudra.WetMaskBlock`` (coupled to the Samudra/OM4 global
zarr conventions via ``credit.ocean.samudra_*``), this block is self-contained
for the regional MOM6 grid: it derives its masks directly from the model's own
static geometry file (``wet`` surface mask + ``deptho`` bathymetry + ``zl``
layer depths).

It operates on the reconstructed output dict written by ``Reconstruct``::

    batch_dict["y_processed"][source][var_key] -> tensor (B, n_levels, n_time, H, W)

For every selected variable the tensor is multiplied by a wet mask (ocean = 1,
land = 0):

* 3D variables (``tensor.shape[1] == n_levels``) use the 3D mask
  ``wet3d[k, y, x] = 1`` iff ``zl[k] <= deptho[y, x]`` and ``wet[y, x] == 1``.
* 2D / surface variables (``tensor.shape[1] == 1``) use the surface mask.

Because ``Reconstruct`` detaches ``y_processed``, this masking shapes the
rollout feed-back and the written outputs, **not** the single-step training
gradient (see OCEAN_MIGRATION_NOTES.md).

Example config::

    postblocks:
      per_step:
        reconstruct: { type: reconstruct }
        wet_mask:
          type: ocean_wet_mask
          args:
            static_path: /.../CREDIT_Input.mom6.h.static.nc
            levels: [ ... ]          # must match the dataset's level subset
"""

from __future__ import annotations

import logging

import xarray as xr
import torch

from credit.postblock.base import BasePostblock

logger = logging.getLogger(__name__)


class OceanWetMask(BasePostblock):
    """Zero land points in reconstructed ocean predictions using static geometry.

    The surface ``wet`` mask and ``deptho`` bathymetry come from ``static_path``.
    Layer depths (``zl``) are **not** in the MOM6 static file — they live in the
    ocean-state file — so they must be supplied via ``level_depths`` (explicit
    list) or ``level_source_path`` (a file to read ``level_var`` from). If
    neither is given, the depth-aware 3D mask is skipped and the surface mask is
    broadcast across all levels of 3D variables (a warning is logged).

    Args:
        static_path: Path to the MOM6 static/geometry NetCDF or Zarr file
            (provides ``wet_var`` and ``deptho_var``).
        wet_var: Name of the 0/1 surface wet mask variable (default ``"wet"``).
        deptho_var: Name of the bathymetry (ocean depth, m) variable (default ``"deptho"``).
        level_depths: Explicit list of layer depths (m, positive down) to build
            the depth-aware 3D mask. Takes precedence over ``level_source_path``.
        level_source_path: File to read layer depths from when ``level_depths``
            is not given (e.g. the ocean-state file that carries ``zl``).
        level_var: Name of the layer-depth coordinate (default ``"zl"``).
        levels: Optional subset of layer depths (nearest-matched) — must match
            the ``levels`` used by the dataset.
        variables: Optional list of ``var_key`` strings to mask. If omitted,
            every variable in ``y_processed`` is masked.
        key: Which ``batch_dict`` entry holds the reconstructed dict
            (default ``"y_processed"``).
    """

    def __init__(
        self,
        static_path: str,
        wet_var: str = "wet",
        deptho_var: str = "deptho",
        level_depths: list | None = None,
        level_source_path: str | None = None,
        level_var: str = "zl",
        levels: list | None = None,
        variables: list[str] | None = None,
        key: str = "y_processed",
    ):
        super().__init__()
        self.variables = variables
        self.key = key

        with xr.open_dataset(static_path) as ds:
            wet = torch.tensor(ds[wet_var].values, dtype=torch.float32)  # (H, W), 0/1
            deptho = torch.tensor(ds[deptho_var].values, dtype=torch.float32)  # (H, W)

        # deptho is NaN over land in MOM6; treat NaN as depth 0 (dry everywhere).
        deptho = torch.nan_to_num(deptho, nan=0.0)
        wet = torch.nan_to_num(wet, nan=0.0)

        zl = self._resolve_level_depths(level_depths, level_source_path, level_var, levels)

        if zl is not None:
            # 3D wet mask: a cell at layer depth zl[k] is wet where the ocean floor
            # is deeper than that layer AND the surface cell is ocean.
            wet3d = ((zl.view(-1, 1, 1) <= deptho.unsqueeze(0)) & (wet.unsqueeze(0) > 0)).float()
            self.n_levels = int(zl.shape[0])
            self.register_buffer("wet_3d", wet3d.unsqueeze(0).unsqueeze(2))  # (1, n_levels, 1, H, W)
        else:
            logger.warning(
                "OceanWetMask: no level_depths or level_source_path given; skipping the "
                "depth-aware 3D mask and broadcasting the surface mask across all levels. "
                "Cells below bathymetry will NOT be masked."
            )
            self.n_levels = None
            self.register_buffer("wet_3d", None)

        # Surface mask, broadcastable to (B, 1, T, H, W)
        self.register_buffer("wet_surface", wet.view(1, 1, 1, *wet.shape))

    @staticmethod
    def _resolve_level_depths(level_depths, level_source_path, level_var, levels):
        """Return a 1-D tensor of layer depths, or None if unavailable."""
        if level_depths is not None:
            zl = torch.tensor(level_depths, dtype=torch.float32)
        elif level_source_path is not None:
            with xr.open_dataset(level_source_path) as ds:
                da = ds[level_var]
                if levels is not None:
                    da = da.sel({level_var: levels}, method="nearest")
                zl = torch.tensor(da.values, dtype=torch.float32)
        else:
            return None
        return zl

    def forward(self, batch_dict: dict) -> dict:
        processed = batch_dict.get(self.key)
        if not isinstance(processed, dict):
            return batch_dict

        for source_vars in processed.values():
            keys = self.variables if self.variables is not None else list(source_vars.keys())
            for var_key in keys:
                if var_key not in source_vars:
                    continue
                tensor = source_vars[var_key]
                n_lev = tensor.shape[1]
                if self.wet_3d is not None and n_lev == self.n_levels:
                    mask = self.wet_3d
                else:
                    # Surface (2D) var, or 3D var when no depth-aware mask is
                    # available / level count differs: broadcast the surface mask.
                    mask = self.wet_surface
                source_vars[var_key] = tensor * mask.to(device=tensor.device, dtype=tensor.dtype)

        return batch_dict
