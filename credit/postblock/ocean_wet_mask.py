"""
ocean_wet_mask.py
-----------------
OceanWetMask: gen2 postblock that blanks land points in ocean predictions.

Unlike ``wet_mask_samudra.WetMaskBlock`` (coupled to the Samudra/OM4 global
zarr conventions via ``credit.ocean.samudra_*``), this block is self-contained
for the regional MOM6 grid: it derives its masks directly from the model's own
static geometry file (``wet`` surface mask + ``deptho`` bathymetry + ``zl``
layer depths).

It operates on the reconstructed output dict written by ``Reconstruct``::

    batch_dict["y_processed"][source][var_key] -> tensor (B, n_levels, n_time, H, W)

For every selected variable, cells outside the wet mask are set to NaN:

* 3D variables (``tensor.shape[1] == n_levels``) use the 3D mask
  ``wet3d[k, y, x] = 1`` iff ``zl[k] <= deptho[y, x]`` and ``wet[y, x] == 1``.
* 2D / surface variables (``tensor.shape[1] == 1``) use the surface mask.

**Why NaN and not 0** (``masked_fill_nan``, default True). ``y_processed`` is in
*physical* units, and in multi-step rollout it is fed back through
``assemble_rollout_batch`` into the preblock chain ``normalize -> fill_values ->
concat``. Writing physical 0.0 makes ``normalize`` map masked cells to
``(0 - mu)/sigma``, which for rMOM6 is -9.9 (thetao, levelwise) to -305,000
(so, pointwise) -- while at step 1 the dataset's land is NaN, which ``normalize``
passes through and ``fill_values`` turns into 0.0 in *normalized* space. Zeroing
here therefore made step 2 of every rollout structurally different from step 1
over land and every below-bathymetry cell, and models trained on that diverged
within one autoregressive step. NaN matches the dataset's own land convention, so
``fill_values`` produces the identical normalized-space 0.0 at every step.

Set ``masked_fill_nan: False`` to restore multiply-by-mask zeroing for a
pipeline whose feedback path stays in normalized space (as the gen1
``wet_mask_samudra.WetMaskBlock`` did) or that has no NaN-filling preblock.

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
import torch.nn.functional as F

from credit.postblock.base import BasePostblock

logger = logging.getLogger(__name__)


class OceanWetMask(BasePostblock):
    """Blank land points in reconstructed ocean predictions using static geometry.

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
        masked_fill_nan: Write NaN into masked cells (default True) instead of 0.
            See the module docstring -- zeroing corrupts the multi-step feedback
            path, because physical 0 does not normalize back to normalized 0.
        halo_pad: ``((lat_start, lat_end), (lon_start, lon_end))`` cells appended to the
            grid by the ``ocean_obc_halo`` preblock. The static geometry is edge-replicated
            to match, so the prescribed-boundary halo is treated as open ocean rather than
            masked away. Must equal the preblock's ``pad_lat``/``pad_lon``.
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
        masked_fill_nan: bool = True,
        halo_pad: tuple = ((0, 0), (0, 0)),
    ):
        super().__init__()
        self.variables = variables
        self.key = key
        self.masked_fill_nan = masked_fill_nan
        self.register_buffer("_nan", torch.tensor(float("nan")))

        with xr.open_dataset(static_path) as ds:
            wet = torch.tensor(ds[wet_var].values, dtype=torch.float32)  # (H, W), 0/1
            deptho = torch.tensor(ds[deptho_var].values, dtype=torch.float32)  # (H, W)

        # deptho is NaN over land in MOM6; treat NaN as depth 0 (dry everywhere).
        deptho = torch.nan_to_num(deptho, nan=0.0)
        wet = torch.nan_to_num(wet, nan=0.0)

        # An OBC halo appended by the ocean_obc_halo preblock makes predictions one cell larger
        # than the static geometry. Extend the masks by edge replication so the halo inherits
        # its neighbour's wet/depth state -- the halo is prescribed open ocean, so masking it
        # dry would blank the boundary condition the model was just handed.
        if any(halo_pad):
            lat0, lat1, lon0, lon1 = (*halo_pad[0], *halo_pad[1]) if len(halo_pad) == 2 else halo_pad
            pad = (lon0, lon1, lat0, lat1)
            wet = F.pad(wet[None, None], pad, mode="replicate")[0, 0]
            deptho = F.pad(deptho[None, None], pad, mode="replicate")[0, 0]
            logger.info("OceanWetMask: extended static geometry by halo_pad=%s -> %s", halo_pad, tuple(wet.shape))

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
                mask = mask.to(device=tensor.device, dtype=tensor.dtype)
                if self.masked_fill_nan:
                    source_vars[var_key] = torch.where(mask > 0, tensor, self._nan.to(tensor.dtype))
                else:
                    source_vars[var_key] = tensor * mask

        return batch_dict
