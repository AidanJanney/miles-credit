"""test_ocean_obc_nudge.py — tests for credit.postblock.ocean_obc_nudge.OceanOBCNudge.

Focus is the vectorized-batch rewrite: OceanOBCNudge now preloads each direction's OBC zarr
store into memory once at construction and, per rollout step, does one vectorized
nearest-timestamp lookup/gather per (variable, direction) covering the whole batch, instead of
a per-batch-element Python loop each issuing its own Zarr read. The single most important test
here is that different batch elements with different (non-sequential) timestamps each get their
own correct nudge values -- the classic bug a naive vectorization introduces is silently
broadcasting one batch element's values across the whole batch.
"""

import numpy as np
import pandas as pd
import torch
import xarray as xr

from credit.postblock.ocean_obc_nudge import OceanOBCNudge

SOURCE = "Test_rMOM6"
THETAO_KEY = f"{SOURCE}/prognostic/3d/thetao"
SSH_KEY = f"{SOURCE}/prognostic/2d/SSH"
SO_KEY = f"{SOURCE}/prognostic/3d/so"

N_TIME = 6
N_Z = 3
H, W = 5, 4  # north/south edge varies along W (xh); east/west edge varies along H (yh)
TIMES = pd.date_range("2000-01-01", periods=N_TIME, freq="D")
# levels chosen so nearest-match to [3.8, 5.1, 7.9] recovers each native level unambiguously
LEVELS = [3.8, 5.1, 7.9]


def _edge_len(direction: str) -> int:
    return W if direction in ("north", "south") else H


def _write_obc_store(tmp_path, direction: str, encode: bool = True, fill: float | None = None, vars_=("thetao", "SSH")):
    """Write a tiny synthetic OBC zarr store for one direction.

    If `encode`, thetao[t, z, x] = t*1000 + z*100 + x and SSH[t, x] = t*1000 + x -- an
    "index-encoding" trick so a wrong-index bug shows up as a wrong number, not a coincidentally
    plausible one. If `fill` is given instead, every value is that constant (used for the
    corner-pixel write-order test, where a simple constant is easier to reason about).
    """
    edge_len = _edge_len(direction)
    data_vars = {}
    if "thetao" in vars_:
        if encode:
            t, z, x = np.meshgrid(np.arange(N_TIME), np.arange(N_Z), np.arange(edge_len), indexing="ij")
            values = (t * 1000 + z * 100 + x).astype("float32")
        else:
            values = np.full((N_TIME, N_Z, edge_len), fill, dtype="float32")
        data_vars["thetao"] = (("time", "z1_l", "edge"), values)
    if "SSH" in vars_:
        if encode:
            t, x = np.meshgrid(np.arange(N_TIME), np.arange(edge_len), indexing="ij")
            values = (t * 1000 + x).astype("float32")
        else:
            values = np.full((N_TIME, edge_len), fill, dtype="float32")
        data_vars["SSH"] = (("time", "edge"), values)
    if "so" in vars_:
        values = np.zeros((N_TIME, N_Z, edge_len), dtype="float32")
        data_vars["so"] = (("time", "z1_l", "edge"), values)

    ds = xr.Dataset(
        data_vars,
        coords={"time": TIMES, "z1_l": np.array([3.8, 5.1, 7.9], dtype="float32")},
    )
    out_path = tmp_path / f"obc_{direction}.zarr"
    ds.to_zarr(out_path, mode="w")
    return str(out_path)


def _make_block(tmp_path, directions, encode=True, fill=None, vars_=("thetao", "SSH"), levels=LEVELS, variables=None):
    obc_paths = {d: _write_obc_store(tmp_path, d, encode=encode, fill=fill, vars_=vars_) for d in directions}
    if variables is None:
        variables = []
        if "thetao" in vars_:
            variables.append(THETAO_KEY)
        if "SSH" in vars_:
            variables.append(SSH_KEY)
        if "so" in vars_:
            variables.append(SO_KEY)
    return OceanOBCNudge(obc_paths=obc_paths, variables=variables, levels=levels)


def _make_batch(var_keys, B, datetimes):
    tensor = {k: torch.zeros(B, N_Z if k != SSH_KEY else 1, 1, H, W) for k in var_keys}
    batch_dict = {
        "y_processed": {SOURCE: tensor},
        "metadata": {"target": {SOURCE: {"datetime": pd.DatetimeIndex(datetimes)}}},
    }
    return batch_dict


def test_batch_elements_get_independent_timestamps(tmp_path):
    """The core regression test: each batch element must get its OWN timestamp's values."""
    block = _make_block(tmp_path, ["north"], vars_=("thetao",))

    # 3 batch elements, deliberately non-sequential and not aligned with batch index.
    chosen_time_idx = [4, 1, 3]
    query_times = TIMES[chosen_time_idx]
    batch_dict = _make_batch([THETAO_KEY], B=3, datetimes=query_times)

    out = block(batch_dict)
    tensor = out["y_processed"][SOURCE][THETAO_KEY]

    edge_idx = -1  # north: last row along H, all W columns
    for b, t_idx in enumerate(chosen_time_idx):
        # levels [3.8, 5.1, 7.9] nearest-match native z1_l indices [0, 1, 2] exactly
        expected = torch.tensor([[t_idx * 1000 + z * 100 + x for x in range(W)] for z in range(N_Z)])
        actual = tensor[b, :, 0, edge_idx, :]
        assert torch.allclose(actual, expected.float()), f"batch element {b} got wrong direction's/timestamp's values"


def test_nearest_neighbor_tie_break_matches_xarray(tmp_path):
    path = _write_obc_store(tmp_path, "north", vars_=("thetao",))
    block = OceanOBCNudge(obc_paths={"north": path}, variables=[THETAO_KEY], levels=LEVELS)

    # Query a timestamp closer to TIMES[2] than TIMES[3] (not exactly halfway, to avoid
    # ambiguous tie-breaking rules differing between implementations).
    query_time = TIMES[2] + pd.Timedelta(hours=1)
    batch_dict = _make_batch([THETAO_KEY], B=1, datetimes=[query_time])
    out = block(batch_dict)
    tensor = out["y_processed"][SOURCE][THETAO_KEY]

    ds = xr.open_zarr(path)
    expected_da = ds["thetao"].sel(z1_l=LEVELS, method="nearest").sel(time=query_time, method="nearest")
    expected = torch.from_numpy(expected_da.values).float()

    actual = tensor[0, :, 0, -1, :]
    assert torch.allclose(actual, expected)


def test_2d_var_broadcast_over_nz(tmp_path):
    block = _make_block(tmp_path, ["south"], vars_=("SSH",))
    batch_dict = _make_batch([SSH_KEY], B=2, datetimes=[TIMES[0], TIMES[5]])
    out = block(batch_dict)
    tensor = out["y_processed"][SOURCE][SSH_KEY]
    assert tensor.shape == (2, 1, 1, H, W)

    for b, t_idx in enumerate([0, 5]):
        expected = torch.tensor([t_idx * 1000 + x for x in range(W)]).float()
        actual = tensor[b, 0, 0, 0, :]  # south: first row (edge_idx=0), all W
        assert torch.allclose(actual, expected)


def test_missing_variable_in_direction_is_skipped(tmp_path):
    # "west" store only has thetao, not "so" -- "so" should be silently skipped for west.
    obc_paths = {
        "west": _write_obc_store(tmp_path, "west", vars_=("thetao",)),
    }
    block = OceanOBCNudge(obc_paths=obc_paths, variables=[SO_KEY], levels=LEVELS)
    batch_dict = _make_batch([SO_KEY], B=1, datetimes=[TIMES[0]])
    original = batch_dict["y_processed"][SOURCE][SO_KEY].clone()
    out = block(batch_dict)
    assert torch.equal(out["y_processed"][SOURCE][SO_KEY], original)  # untouched, no raise


def test_missing_datetime_is_noop(tmp_path):
    block = _make_block(tmp_path, ["north"], vars_=("thetao",))
    tensor = torch.randn(2, N_Z, 1, H, W)
    batch_dict = {
        "y_processed": {SOURCE: {THETAO_KEY: tensor.clone()}},
        "metadata": {"target": {SOURCE: {}}},  # no "datetime" key
    }
    out = block(batch_dict)
    assert torch.equal(out["y_processed"][SOURCE][THETAO_KEY], tensor)


def test_corner_pixel_write_order_preserved(tmp_path):
    """North (whole top row) and west (whole left column) both touch the NW corner pixel.

    forward() iterates directions in construction order, so whichever direction is inserted
    last into obc_paths wins the shared corner -- this must match today's dict-iteration-order
    behavior exactly.
    """
    obc_paths = {
        "north": _write_obc_store(tmp_path, "north", encode=False, fill=100.0, vars_=("thetao",)),
        "west": _write_obc_store(tmp_path, "west", encode=False, fill=200.0, vars_=("thetao",)),
    }
    block = OceanOBCNudge(obc_paths=obc_paths, variables=[THETAO_KEY], levels=LEVELS)
    batch_dict = _make_batch([THETAO_KEY], B=1, datetimes=[TIMES[0]])
    out = block(batch_dict)
    tensor = out["y_processed"][SOURCE][THETAO_KEY]

    # NW corner: last row (north, edge_idx=-1), first column (west, edge_idx=0).
    corner = tensor[0, :, 0, -1, 0]
    assert torch.allclose(corner, torch.full((N_Z,), 200.0)), "west (inserted last) should win the shared corner"

    # A non-corner point on the north edge should still be north's value.
    non_corner_north = tensor[0, :, 0, -1, 1]
    assert torch.allclose(non_corner_north, torch.full((N_Z,), 100.0))


def test_levels_preselection_matches_reference(tmp_path):
    path = _write_obc_store(tmp_path, "east", vars_=("thetao",))
    # requested depths don't exactly match the native z1_l values [3.8, 5.1, 7.9] -- each
    # should nearest-match to the same native level (and thus the same encoded values) as if
    # the exact native depths had been requested.
    requested_levels = [3.85, 5.05, 7.85]
    block = OceanOBCNudge(obc_paths={"east": path}, variables=[THETAO_KEY], levels=requested_levels)
    batch_dict = _make_batch([THETAO_KEY], B=1, datetimes=[TIMES[3]])
    out = block(batch_dict)
    tensor = out["y_processed"][SOURCE][THETAO_KEY]

    ds = xr.open_zarr(path)
    expected_da = ds["thetao"].sel(z1_l=requested_levels, method="nearest").sel(time=TIMES[3], method="nearest")
    expected = torch.from_numpy(expected_da.values).float()

    # east: edge_idx=-1, axis=-1 -> tensor[:, :, 0, :, -1]
    actual = tensor[0, :, 0, :, -1]
    assert actual.shape[0] == N_Z
    assert torch.allclose(actual, expected)
