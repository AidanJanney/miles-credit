"""
ocean_obc_nudge.py
------------------
OceanOBCNudge: a naive open-boundary-condition (OBC) treatment for the regional ocean
emulator's autoregressive rollout.

Every rollout step, the model runs on a doubly-periodic-shaped tensor with no notion of what
lies outside the domain -- nothing in the training data currently tells it what the true ocean
state is doing at the lateral edges (see OCEAN_MIGRATION_NOTES.md, "OBCs are not currently
connected to the pipeline"). This block is the cheapest possible fix: it literally overwrites
the outer single-pixel-wide ring of each nudged prognostic variable with the prescribed OBC
value at the current step's real timestamp, read directly from the zarr stores
``scripts/preprocess_obcs.py`` produces. It is **not** a proper sponge layer (no smooth
blending into the interior, no relaxation timescale) and does **not** feed OBCs into the model
as an input the way a real forcing channel would -- see the "dedicated OBC input pathway"
follow-up in OCEAN_MIGRATION_NOTES.md for the two real designs this is standing in for.

Runs in ``postblocks.per_step``, after ``ocean_wet_mask`` (needs physical units and a
land-masked field first) and before ``post_rollout`` blocks. Requires ``batch_dict["metadata"]
["target"][source]["datetime"]`` to know the current step's real calendar time -- this is only
reliably present at every rollout step (not just step 0) since the ``assemble_rollout_batch``
fix in ``credit/trainers/rollout_utils.py`` (forwards ``curr_batch["metadata"]`` through; it
used to be silently dropped after step 0). If that key is missing for a given source, this
block is a no-op for that source rather than raising -- the rest of the postblock chain still
completes.

y_processed tensors returned by ``Reconstruct`` are *views* into ``y_pred``'s storage
(``y_pred[:, ch_slice, ...].detach()`` -- ``detach()`` does not copy memory), so in-place
edits would corrupt the still-attached ``y_pred`` used for the training loss. This block always
``.clone()``s before writing into the boundary ring, matching the non-in-place pattern every
other postblock (``ocean_clamp``'s ``torch.clamp`` returns a new tensor) already follows.

Performance note: the OBC zarr stores are boundary rings only (one edge's worth of pixels, not
the full domain), so the entire time series for every configured direction/variable is preloaded
into CPU memory once at construction (``__init__``) rather than read lazily per rollout step.
``forward()`` then does one vectorized nearest-timestamp lookup and gather per (variable,
direction) covering the whole batch at once, instead of a per-batch-element, per-direction loop
each issuing its own Zarr read -- the original per-call ``xr.Dataset.sel(...)`` pattern scaled as
``batch_size x n_vars x n_directions x forecast_len`` individual disk reads per training batch,
executed synchronously in the training loop.
"""

from __future__ import annotations

from os.path import expandvars

import pandas as pd
import torch
import xarray as xr

from credit.postblock.base import BasePostblock

_DIRECTIONS = ("north", "south", "east", "west")
# direction -> (edge index, spatial axis position in (nz, H, W): -2 is H, -1 is W)
_EDGE_SPEC = {
    "north": (-1, -2),  # last row along H (yh)
    "south": (0, -2),  # first row along H (yh)
    "east": (-1, -1),  # last column along W (xh)
    "west": (0, -1),  # first column along W (xh)
}


class OceanOBCNudge(BasePostblock):
    """Overwrite the domain-edge ring of ``y_processed`` with prescribed OBC values.

    Args:
        obc_paths: mapping of direction -> zarr path, e.g.
            ``{"north": ".../obc_north_sample.zarr", "south": ..., "east": ..., "west": ...}``.
            Any subset of the 4 directions may be given; missing ones are skipped.
        variables: full var_keys to nudge (e.g. ``"rMOM6/prognostic/3d/thetao"``). The last
            path segment (e.g. ``"thetao"``) must match a data variable in the OBC zarrs
            (``thetao``, ``so``, ``uo``, ``vo``, ``SSH`` -- the canonical names
            ``scripts/preprocess_obcs.py`` renames to).
        levels: comma-separated-style list of depths (metres), nearest-matched against the OBC
            zarr's ``z1_l`` coordinate, restricting to the same vertical-level subset the model
            was trained on (e.g. the 25 autoencoder-selected depths). ``None`` = use every
            level in the OBC file (only correct if the model itself uses all levels).
        key: ``batch_dict`` entry holding the reconstructed dict (default ``"y_processed"``).

    Config example::

        type: "ocean_obc_nudge"
        args:
            obc_paths:
                north: "/path/to/obc_north_sample.zarr"
                south: "/path/to/obc_south_sample.zarr"
                east: "/path/to/obc_east_sample.zarr"
                west: "/path/to/obc_west_sample.zarr"
            variables:
                - "rMOM6/prognostic/3d/thetao"
                - "rMOM6/prognostic/3d/so"
                - "rMOM6/prognostic/3d/uo"
                - "rMOM6/prognostic/3d/vo"
                - "rMOM6/prognostic/2d/SSH"
            levels: [3.8, 5.1, 7.9, 9.6, 92.3, ...]   # match model.levels exactly
    """

    def __init__(
        self,
        obc_paths: dict[str, str],
        variables: list[str],
        levels: list[float] | None = None,
        key: str = "y_processed",
    ):
        super().__init__()
        self.key = key
        self.variables = variables
        self.levels = levels

        # Preloaded once, keyed by (direction, varname) -> full (time, [z1_l], edge_len) tensor.
        # Iterate obc_paths.items() in its given order so any corner pixel shared by two
        # directions (e.g. north overwrites the whole top row, west the whole left column --
        # both touch the NW corner) keeps the same "last direction wins" behavior forward()'s
        # own iteration order previously produced.
        self._obc_data: dict[tuple[str, str], torch.Tensor] = {}
        self._obc_time_idx: dict[str, pd.DatetimeIndex] = {}
        for direction, path in obc_paths.items():
            if direction not in _DIRECTIONS:
                continue
            ds = xr.open_dataset(expandvars(path), engine="zarr")
            self._obc_time_idx[direction] = pd.DatetimeIndex(ds["time"].values)
            for var_key in variables:
                varname = var_key.split("/")[-1]
                if varname not in ds:
                    continue
                da = ds[varname]
                if self.levels is not None and "z1_l" in da.dims:
                    da = da.sel(z1_l=self.levels, method="nearest")
                self._obc_data[(direction, varname)] = torch.from_numpy(da.values)
            ds.close()

    def forward(self, batch_dict: dict) -> dict:
        processed = batch_dict.get(self.key)
        if not isinstance(processed, dict):
            return batch_dict
        metadata = batch_dict.get("metadata") or {}

        for source, source_vars in processed.items():
            dt_index = metadata.get("target", {}).get(source, {}).get("datetime")
            if dt_index is None:
                continue  # no timestamp available for this source -- skip nudging, don't raise

            for var_key in self.variables:
                if var_key not in source_vars:
                    continue
                varname = var_key.split("/")[-1]
                tensor = source_vars[var_key].clone()  # (B, nz, T=1, H, W); never mutate the original view
                B = tensor.shape[0]

                # Same clamping behavior as the original per-batch-element `min(b, len-1)`:
                # if fewer timestamps than batch elements are available, repeat the last one.
                n = len(dt_index)
                if n >= B:
                    query_times = dt_index[:B]
                else:
                    query_times = dt_index[list(range(n)) + [n - 1] * (B - n)]

                for direction, time_idx in self._obc_time_idx.items():
                    arr = self._obc_data.get((direction, varname))
                    if arr is None:
                        continue
                    edge_idx, axis = _EDGE_SPEC[direction]

                    idx = torch.as_tensor(time_idx.get_indexer(query_times, method="nearest"), dtype=torch.long)
                    vals = arr[idx]  # (B, nz, edge_len) for 3D vars, (B, edge_len) for 2D
                    vals = vals.to(dtype=tensor.dtype, device=tensor.device)
                    if vals.dim() == 2:  # 2D var (e.g. SSH): broadcast over the nz=1 axis
                        vals = vals.unsqueeze(1)
                    if axis == -2:  # north/south: overwrite a full row (all levels, all W)
                        tensor[:, :, 0, edge_idx, :] = vals
                    else:  # east/west: overwrite a full column (all levels, all H)
                        tensor[:, :, 0, :, edge_idx] = vals

                source_vars[var_key] = tensor

        return batch_dict
