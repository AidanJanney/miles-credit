"""Regression test for zero-filled face seams on the cubed-sphere grid.

``CubedWXFormer.se_to_cube`` scatters ``ncol`` SE nodes into a dense
``6 x E x E`` cube.  The SE grid de-duplicates cells shared between faces, so
some cube cells have no owning SE node and stay at the ``new_zeros`` fill.  On
ne120 that is 4,324 cells per channel, all of them on face seams (faces 2/3
lose two edge rings each, the two polar faces 4/5 lose all four).

``HaloExchange`` is what is supposed to repair them: it reprojects every padded
coordinate back to a physically equivalent SE-owned cell.  This test pins the
property that matters for training -- after the pad step, nothing inside the
native face window is still sitting at the scatter's zero fill.

Left broken, those cells feed a ring of "zero" (i.e. climatological mean, in
normalized units) into the encoder along every face seam at every step, which
is a compounding artifact source under autoregressive rollout.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from credit.models.wxformer.cubed_wxformer import NFACE
from credit.models.wxformer.halo import HaloExchange, NFACE_EDGE


def _static_dir():
    return Path(
        os.environ.get(
            "MESACLIP_STATIC",
            Path(__file__).resolve().parents[2] / "credit-mesaclip" / "mesaclip" / "static",
        )
    )


def test_halo_leaves_no_unfilled_cells_in_the_native_face():
    """Every SE-unowned cell inside the native face is filled from a neighbour."""
    static = _static_dir()
    se_index_path = static / "se_index_ne120.npy"
    adjacency_path = static / "se_face_adjacency_ne120.npz"
    if not se_index_path.exists() or not adjacency_path.exists():
        pytest.skip("ne120 cubed-sphere static files are not available")

    crop = 11
    halo = HaloExchange(
        adjacency_path=adjacency_path,
        se_index_path=se_index_path,
        padded_size=384,
        crop_top=crop,
        crop_left=crop,
    )

    # Mirror CubedWXFormer.se_to_cube: scatter a constant field onto the owned
    # SE nodes and leave every unowned cube cell at the zero fill, so any cell
    # the halo fails to repair shows up as an exact 0.0 against a constant 1.0.
    se_index = torch.from_numpy(np.load(se_index_path).astype(np.int64))
    cube = torch.zeros(1, 1, NFACE * NFACE_EDGE * NFACE_EDGE)
    cube[:, :, se_index] = 1.0
    x6 = (
        cube.reshape(1, 1, NFACE, NFACE_EDGE, NFACE_EDGE)
        .permute(0, 2, 1, 3, 4)
        .reshape(NFACE, 1, NFACE_EDGE, NFACE_EDGE)
    )

    padded = halo(x6)
    native = padded[:, :, crop : crop + NFACE_EDGE, crop : crop + NFACE_EDGE]

    unfilled = int((native == 0.0).sum())
    assert unfilled == 0, (
        f"{unfilled} cells inside the native face window are still at the "
        f"scatter's zero fill after HaloExchange "
        f"(per face: {[int((native[f] == 0.0).sum()) for f in range(NFACE)]}). "
        "These are the SE-deduplicated face-seam cells; they must be gathered "
        "from the owning neighbour face, not left at zero."
    )
    torch.testing.assert_close(native, torch.ones_like(native))
