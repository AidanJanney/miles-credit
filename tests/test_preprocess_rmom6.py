"""test_preprocess_rmom6.py — unit tests for the Zarr chunking behavior in
scripts/preprocess_rmom6.py and scripts/preprocess_obcs.py.

Covers the fix that stopped chunking the vertical level dimension (``z1_l``) to size 1 --
level subsetting already happens once at preprocessing time via ``--levels``, and every
training step reads all of a variable's levels for its timestep, so a separate Zarr chunk per
level only multiplied per-step chunk-read fan-out for no benefit.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import xarray as xr

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


preprocess_rmom6 = _load_module("preprocess_rmom6", "preprocess_rmom6.py")
preprocess_obcs = _load_module("preprocess_obcs", "preprocess_obcs.py")


def _make_synthetic_prognostic_dataset(n_time=10, n_z=5, n_yh=6, n_xh=7) -> xr.Dataset:
    return xr.Dataset(
        {
            "thetao": (("time", "z1_l", "yh", "xh"), np.random.randn(n_time, n_z, n_yh, n_xh).astype("float32")),
            "SSH": (("time", "yh", "xh"), np.random.randn(n_time, n_yh, n_xh).astype("float32")),
        }
    )


def test_write_zarr_z1_l_chunk_is_full_size_not_one(tmp_path):
    ds = _make_synthetic_prognostic_dataset(n_time=10, n_z=5, n_yh=6, n_xh=7)
    time_chunk = 4
    out_path = tmp_path / "prognostic.zarr"

    preprocess_rmom6._write_zarr(ds, str(out_path), time_chunk, label="test")

    reopened = xr.open_zarr(out_path)
    thetao_chunks = reopened["thetao"].encoding["chunks"]
    # dims are (time, z1_l, yh, xh) -- z1_l chunk size must equal the full dimension size (5),
    # not 1.
    assert thetao_chunks == (min(time_chunk, 10), 5, 6, 7)


def test_write_zarr_forcing_store_unaffected(tmp_path):
    """Forcing store has no z1_l dim -- confirm the change doesn't touch it."""
    ds = xr.Dataset({"taux": (("time", "yh", "xh"), np.random.randn(10, 6, 7).astype("float32"))})
    time_chunk = 4
    out_path = tmp_path / "forcing.zarr"

    preprocess_rmom6._write_zarr(ds, str(out_path), time_chunk, label="test")

    reopened = xr.open_zarr(out_path)
    assert reopened["taux"].encoding["chunks"] == (min(time_chunk, 10), 6, 7)


def test_write_zarr_round_trip_values_correct(tmp_path):
    ds = _make_synthetic_prognostic_dataset(n_time=10, n_z=5, n_yh=6, n_xh=7)
    out_path = tmp_path / "prognostic.zarr"

    preprocess_rmom6._write_zarr(ds, str(out_path), 4, label="test")

    reopened = xr.open_zarr(out_path)
    np.testing.assert_allclose(reopened["thetao"].values, ds["thetao"].values)
    np.testing.assert_allclose(reopened["SSH"].values, ds["SSH"].values)


def test_layer_interfaces_reconstruct_centers():
    z1_l = np.array([0.49, 1.54, 2.65, 3.82, 5.08, 6.44, 7.93, 9.57])
    z1_i = preprocess_rmom6._layer_interfaces(z1_l)

    assert z1_i[0] == 0.0
    assert len(z1_i) == len(z1_l) + 1
    # a layer's center is by construction the midpoint of its bracketing interfaces
    np.testing.assert_allclose(0.5 * (z1_i[:-1] + z1_i[1:]), z1_l)


def test_pairwise_thickness_average_odd_levels_raises():
    ds = _make_synthetic_prognostic_dataset(n_time=3, n_z=5, n_yh=2, n_xh=2)
    ds = ds.assign_coords(z1_l=np.arange(5, dtype=float))
    try:
        preprocess_rmom6.pairwise_thickness_average(ds)
        assert False, "expected ValueError on odd level count"
    except ValueError:
        pass


def test_pairwise_thickness_average_matches_hand_computed_weights():
    z1_l = np.array([0.49, 1.54, 2.65, 3.82, 5.08, 6.44, 7.93, 9.57])
    n = len(z1_l)
    data = np.arange(n, dtype="float32").reshape(1, n, 1, 1) * np.ones((1, n, 2, 2), dtype="float32")
    ds = xr.Dataset({"thetao": (("time", "z1_l", "yh", "xh"), data)}, coords={"z1_l": z1_l})

    out = preprocess_rmom6.pairwise_thickness_average(ds)

    assert out.sizes["z1_l"] == n // 2
    dz = np.diff(preprocess_rmom6._layer_interfaces(z1_l))
    for pair in range(n // 2):
        lo, hi = 2 * pair, 2 * pair + 1
        expected = (lo * dz[lo] + hi * dz[hi]) / (dz[lo] + dz[hi])
        np.testing.assert_allclose(out["thetao"].isel(time=0, z1_l=pair, yh=0, xh=0).values, expected)
        expected_z1_l = (z1_l[lo] * dz[lo] + z1_l[hi] * dz[hi]) / (dz[lo] + dz[hi])
        np.testing.assert_allclose(out["z1_l"].values[pair], expected_z1_l)


def test_pairwise_thickness_average_constant_field_preserved():
    z1_l = np.array([0.49, 1.54, 2.65, 3.82, 5.08, 6.44])
    ds = xr.Dataset(
        {"thetao": (("time", "z1_l", "yh", "xh"), np.full((2, len(z1_l), 3, 3), 7.0, dtype="float32"))},
        coords={"z1_l": z1_l},
    )

    out = preprocess_rmom6.pairwise_thickness_average(ds)

    np.testing.assert_allclose(out["thetao"].values, 7.0)


def test_obc_write_zarr_z1_l_chunk_is_full_size_not_one(tmp_path):
    ds = xr.Dataset(
        {
            "thetao": (("time", "z1_l", "edge"), np.random.randn(8, 4, 5).astype("float32")),
        }
    )
    out_path = tmp_path / "obc_north.zarr"

    preprocess_obcs._write_zarr(ds, str(out_path), time_chunk=3, label="test")

    reopened = xr.open_zarr(out_path)
    assert reopened["thetao"].encoding["chunks"] == (3, 4, 5)
