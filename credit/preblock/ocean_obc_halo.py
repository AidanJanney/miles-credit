"""
ocean_obc_halo.py
-----------------
OceanOBCHalo: gen2 preblock that appends prescribed open-boundary conditions to the model
grid as a one-cell halo, so the network *sees* the boundary as input instead of having its
prediction overwritten there afterwards.

Why a preblock and not a postblock. ``ocean_obc_nudge`` (postblock) overwrites the outer ring
of ``y_processed`` after the forward pass. The model therefore never receives boundary
information: it is scored on predicting the edge, then corrected there, and because postblocks
are detached from the gradient it cannot learn from the correction either. For a regional
domain whose interior variability is largely boundary-driven (the Caribbean Current entering
through the eastern passages), that caps achievable skill. This block instead makes the
boundary an *input*, which is what the gen1 configuration did via ``image_height: 458 # 457 + 1
obc``.

Why the halo is built here rather than in the dataset. Each OBC ring is declared as an ordinary
gen2 source, so it flows through the per-source ``normalize`` preblock and is standardized with
**its own statistics** before this block runs -- boundary cells have different distributions
from the interior, and ``ConcatToTensor`` joins sources along the channel dim (``dim=1``), so a
separate source alone would add channels rather than grid cells. Running after ``normalize``
and before ``concat`` is what lets "normalize each source separately, then concatenate" also
mean "then place into the halo".

What it does, in order:

1. Spatially pads every remaining source's variables from ``(H, W)`` to ``(H + 1, W + 1)``
   (configurable via ``pad_lat``/``pad_lon``), because ``torch.cat(..., dim=1)`` requires all
   sources to share spatial dims. Interior values are copied; halo cells are filled by edge
   replication for variables with no boundary data (forcing, statics).
2. Writes each normalized OBC ring into its edge of the padded prognostic variables.
3. Deletes the OBC sources from the batch so ``concat`` does not also emit them as channels.

Edge orientation is explicit in config rather than inferred. For the rMOM6 Caribbean grid
``yh`` increases northward and ``xh`` eastward, so ``north`` is the last row and ``east`` the
last column; ``obc_south``/``obc_west`` are entirely land and are not used.

The corner cell belongs to two rings. ``corner`` selects which one wins (default ``"lon"``,
i.e. the east ring, whose correlation with the adjacent interior column is far higher than
north's with its row on this grid: 0.98 vs 0.74).

Example config::

    preblocks:
      per_step:
        normalize:   { ... }        # runs per source, OBC sources get their own stats
        fill_land:   { ... }
        obc_halo:
          type: ocean_obc_halo
          args:
            interior_source: rMOM6
            pad_lat: [0, 1]         # [start, end] rows added, same convention as padding_conf
            pad_lon: [0, 1]
            edges:
              rMOM6_obc_north: { source: rMOM6_obc_north, edge: lat_end }
              rMOM6_obc_east:  { source: rMOM6_obc_east,  edge: lon_end }
        concat:      { type: concat }
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from credit.preblock.base import BasePreblock

logger = logging.getLogger(__name__)

_VALID_EDGES = {"lat_start", "lat_end", "lon_start", "lon_end"}


class OceanOBCHalo(BasePreblock):
    """Append OBC rings to the grid as a one-cell halo, before channel concatenation.

    Args:
        interior_source: Name of the source holding the model state to be extended.
        edges: Mapping of a label -> ``{"source": <obc source name>, "edge": <lat_end|...>}``.
        pad_lat: ``[start, end]`` rows to add to the interior grid.
        pad_lon: ``[start, end]`` columns to add.
        corner: ``"lat"`` or ``"lon"`` -- which ring owns a cell claimed by both.
        data_types: Which top-level data types to process (default ``["input"]``). The target
            is left at the interior size unless listed, so the loss keeps its native grid; when
            the target *is* padded, exclude the halo via ``loss.exclude_border``.
    """

    def __init__(
        self,
        interior_source: str,
        edges: dict,
        pad_lat: list[int] | tuple[int, int] = (0, 1),
        pad_lon: list[int] | tuple[int, int] = (0, 1),
        corner: str = "lon",
        data_types: list[str] | None = None,
    ):
        super().__init__()
        self.interior_source = interior_source
        self.edges = edges or {}
        self.pad_lat = tuple(pad_lat)
        self.pad_lon = tuple(pad_lon)
        if corner not in ("lat", "lon"):
            raise ValueError(f"corner must be 'lat' or 'lon', got {corner!r}")
        self.corner = corner
        self.data_types = data_types or ["input"]

        for label, spec in self.edges.items():
            edge = spec.get("edge")
            if edge not in _VALID_EDGES:
                raise ValueError(f"edge {label!r}: 'edge' must be one of {sorted(_VALID_EDGES)}, got {edge!r}")
            if not spec.get("source"):
                raise ValueError(f"edge {label!r}: missing 'source'.")
            need = self.pad_lat if edge.startswith("lat") else self.pad_lon
            slot = need[0] if edge.endswith("start") else need[1]
            if slot < 1:
                raise ValueError(
                    f"edge {label!r} targets {edge}, but the corresponding pad "
                    f"({'pad_lat' if edge.startswith('lat') else 'pad_lon'}={need}) allocates no cell there."
                )

    def _pad(self, tensor: torch.Tensor) -> torch.Tensor:
        """Edge-replicate a (..., H, W) tensor into the halo. F.pad's last-dim-first order."""
        lat0, lat1 = self.pad_lat
        lon0, lon1 = self.pad_lon
        if not (lat0 or lat1 or lon0 or lon1):
            return tensor
        flat = tensor.reshape(-1, 1, tensor.shape[-2], tensor.shape[-1])
        # 'replicate' carries the adjacent interior value outward, so forcing/static channels
        # get a plausible halo value rather than a zero cliff the conv would see as an edge.
        out = F.pad(flat, (lon0, lon1, lat0, lat1), mode="replicate")
        return out.reshape(*tensor.shape[:-2], out.shape[-2], out.shape[-1])

    def _write_ring(self, tensor: torch.Tensor, ring: torch.Tensor, edge: str) -> torch.Tensor:
        """Write a boundary ring into the halo row/column of an already-padded tensor."""
        # ring arrives as (B, n_levels, T, N) or (B, n_levels, T, 1, N) / (..., N, 1)
        ring = ring.squeeze(-1) if ring.shape[-1] == 1 and ring.dim() > tensor.dim() - 1 else ring
        ring = ring.reshape(*ring.shape[: tensor.dim() - 2], -1)

        lat0, lat1 = self.pad_lat
        lon0, lon1 = self.pad_lon
        H, W = tensor.shape[-2], tensor.shape[-1]
        # A ring spans only the interior extent unless it also owns the corner.
        if edge.startswith("lat"):
            row = lat0 - 1 if edge == "lat_start" else H - lat1
            inner_lo, inner_hi = lon0, W - lon1
            lo, hi = (0, W) if self.corner == "lat" else (inner_lo, inner_hi)
            ring = self._extend_to_span(ring, inner_hi - inner_lo, inner_lo - lo, hi - inner_hi, edge)
            tensor = tensor.clone()
            tensor[..., row, lo:hi] = ring.to(tensor.dtype)
        else:
            col = lon0 - 1 if edge == "lon_start" else W - lon1
            inner_lo, inner_hi = lat0, H - lat1
            lo, hi = (0, H) if self.corner == "lon" else (inner_lo, inner_hi)
            ring = self._extend_to_span(ring, inner_hi - inner_lo, inner_lo - lo, hi - inner_hi, edge)
            tensor = tensor.clone()
            tensor[..., lo:hi, col] = ring.to(tensor.dtype)
        return tensor

    @staticmethod
    def _extend_to_span(ring: torch.Tensor, inner_span: int, extra_start: int, extra_end: int, edge: str):
        """Replicate the ring's end values into the corner cells it owns.

        The prescribed boundary files cover the *interior* extent of their edge (the rMOM6 east
        ring is 457 cells for a 457-row interior). When ``corner`` gives this ring the cell where
        two edges meet, the write span is the interior extent plus the halo cells on the
        perpendicular axis, and no boundary file supplies a value for them -- a corner is not on
        either segment. Carrying the ring's terminal value outward is the same edge-replication
        ``_pad`` already applies to forcing and statics, and covers a single cell.
        """
        if ring.shape[-1] != inner_span:
            raise ValueError(f"OBC ring length {ring.shape[-1]} does not match interior span {inner_span} for {edge}.")
        if extra_start or extra_end:
            ring = F.pad(ring.reshape(-1, 1, ring.shape[-1]), (extra_start, extra_end), mode="replicate").reshape(
                *ring.shape[:-1], inner_span + extra_start + extra_end
            )
        return ring

    def forward(self, batch: dict) -> dict:
        batch = self._copy_batch(batch)
        obc_sources = {spec["source"] for spec in self.edges.values()}
        seen_rings: set[str] = set()

        for data_type in self.data_types:
            sources = batch.get(data_type)
            if not isinstance(sources, dict):
                continue

            # 1. Pad every non-OBC source to the halo grid so concat's dim=1 cat is well posed.
            for source, source_vars in sources.items():
                if source in obc_sources:
                    continue
                sources[source] = {k: self._pad(v) for k, v in source_vars.items()}

            # 2. Write each normalized ring into its edge of the interior source.
            interior = sources.get(self.interior_source)
            if interior is None:
                raise KeyError(
                    f"OceanOBCHalo: interior_source {self.interior_source!r} not in "
                    f"{data_type!r} (have {sorted(sources)})."
                )
            for label, spec in self.edges.items():
                ring_vars = sources.get(spec["source"])
                if ring_vars is None:
                    # An OBC ring is declared as dynamic_forcing, so it exists under "input"
                    # and legitimately has nothing under "target". Listing "target" in
                    # data_types asks for the target grid to be *padded* to match the input's
                    # halo, not for rings to be written into it -- those cells are prescribed
                    # and loss.exclude_border drops them anyway. Missing from every data type
                    # is still a real misconfiguration, checked after the loop.
                    continue
                seen_rings.add(spec["source"])
                for ring_key, ring in ring_vars.items():
                    varname = ring_key.split("/")[-1]
                    matches = [k for k in interior if k.split("/")[-1] == varname]
                    if not matches:
                        logger.warning(
                            "OceanOBCHalo: %r has no counterpart in %s; its halo keeps the replicated interior value.",
                            ring_key,
                            self.interior_source,
                        )
                        continue
                    for target_key in matches:
                        interior[target_key] = self._write_ring(interior[target_key], ring, spec["edge"])

            # 3. Drop the OBC sources so concat does not emit them as extra channels.
            for source in obc_sources:
                sources.pop(source, None)

        unseen = obc_sources - seen_rings
        if unseen:
            raise KeyError(
                f"OceanOBCHalo: OBC source(s) {sorted(unseen)} not found in any of "
                f"{self.data_types}. They must be declared in data.source so they are loaded "
                "and normalized before this block."
            )
        return batch
