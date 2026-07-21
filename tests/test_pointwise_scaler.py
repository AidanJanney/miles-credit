"""test_pointwise_scaler.py — tests for the pointwise (per-level, per-pixel) scaler blocks.

Covers credit.preblock.pointwise_stats.{apply_pointwise_scale, get_cached_stat_entry} and the
device-caching behavior of credit.preblock.pointwise_scaler.PointwiseScalerTransform /
credit.postblock.pointwise_scaler.PointwiseScalerTransform.
"""

import torch

from credit.preblock.pointwise_stats import apply_pointwise_scale, get_cached_stat_entry
from credit.preblock.pointwise_scaler import PointwiseScalerTransform as PreScaler
from credit.postblock.pointwise_scaler import PointwiseScalerTransform as PostScaler

VAR_KEY = "Test_rMOM6/prognostic/3d/thetao"
NZ, H, W = 4, 5, 6


def _make_stats_file(tmp_path):
    stats = {
        VAR_KEY: {
            "mu": torch.randn(NZ, H, W),
            "sigma": torch.rand(NZ, H, W) + 0.5,  # avoid near-zero sigma
        }
    }
    path = tmp_path / "stats.pt"
    torch.save(stats, path)
    return str(path)


def test_transform_inverse_transform_round_trip():
    stat_entry = {"mu": torch.randn(NZ, H, W), "sigma": torch.rand(NZ, H, W) + 0.5}
    x = torch.randn(2, NZ, 1, H, W)
    scaled = apply_pointwise_scale(x, stat_entry, "transform")
    recovered = apply_pointwise_scale(scaled, stat_entry, "inverse_transform")
    assert torch.allclose(x, recovered, atol=1e-5)


def test_get_cached_stat_entry_reuses_same_tensor_object():
    stat_entry = {"mu": torch.randn(NZ, H, W), "sigma": torch.rand(NZ, H, W) + 0.5}
    cache: dict = {}
    first = get_cached_stat_entry(cache, stat_entry, VAR_KEY, torch.float32, torch.device("cpu"))
    second = get_cached_stat_entry(cache, stat_entry, VAR_KEY, torch.float32, torch.device("cpu"))
    assert first["mu"] is second["mu"]
    assert first["sigma"] is second["sigma"]
    assert len(cache) == 1


def test_get_cached_stat_entry_separates_by_dtype():
    stat_entry = {"mu": torch.randn(NZ, H, W), "sigma": torch.rand(NZ, H, W) + 0.5}
    cache: dict = {}
    f32 = get_cached_stat_entry(cache, stat_entry, VAR_KEY, torch.float32, torch.device("cpu"))
    f16 = get_cached_stat_entry(cache, stat_entry, VAR_KEY, torch.float16, torch.device("cpu"))
    assert len(cache) == 2
    assert f32["mu"].dtype == torch.float32
    assert f16["mu"].dtype == torch.float16
    assert torch.allclose(f32["mu"], f16["mu"].float(), atol=1e-3)


def test_preblock_forward_matches_uncached_reference(tmp_path):
    stats_path = _make_stats_file(tmp_path)
    block = PreScaler(stats_path=stats_path, variables=[VAR_KEY], method="transform")
    stats = torch.load(stats_path, map_location="cpu")

    batch = {"input": {"Test_rMOM6": {VAR_KEY: torch.randn(3, NZ, 1, H, W)}}}
    x_orig = batch["input"]["Test_rMOM6"][VAR_KEY].clone()

    out = block(batch)
    result = out["input"]["Test_rMOM6"][VAR_KEY]
    reference = apply_pointwise_scale(x_orig, stats[VAR_KEY], "transform")
    assert torch.allclose(result, reference, atol=1e-5)


def test_preblock_forward_reuses_cache_across_calls(tmp_path):
    stats_path = _make_stats_file(tmp_path)
    block = PreScaler(stats_path=stats_path, variables=[VAR_KEY], method="transform")

    batch1 = {"input": {"Test_rMOM6": {VAR_KEY: torch.randn(3, NZ, 1, H, W)}}}
    block(batch1)
    assert len(block._device_cache) == 1
    cached_mu_id = id(next(iter(block._device_cache.values()))["mu"])

    batch2 = {"input": {"Test_rMOM6": {VAR_KEY: torch.randn(3, NZ, 1, H, W)}}}
    block(batch2)
    assert len(block._device_cache) == 1  # no new entry for the same (var_key, device, dtype)
    assert id(next(iter(block._device_cache.values()))["mu"]) == cached_mu_id


def test_postblock_pre_post_round_trip(tmp_path):
    stats_path = _make_stats_file(tmp_path)
    pre = PreScaler(stats_path=stats_path, variables=[VAR_KEY], method="transform")
    post = PostScaler(stats_path=stats_path, variables=[], method="inverse_transform")

    x_orig = torch.randn(2, NZ, 1, H, W)
    batch = {"input": {"Test_rMOM6": {VAR_KEY: x_orig.clone()}}}
    batch = pre(batch)

    batch_dict = {"y_processed": {"Test_rMOM6": {VAR_KEY: batch["input"]["Test_rMOM6"][VAR_KEY]}}}
    result = post(batch_dict)["y_processed"]["Test_rMOM6"][VAR_KEY]

    assert torch.allclose(x_orig, result, atol=1e-4)


def test_missing_var_key_skipped_and_not_cached(tmp_path):
    stats_path = _make_stats_file(tmp_path)
    other_var = "Test_rMOM6/prognostic/3d/other"
    block = PreScaler(stats_path=stats_path, variables=[other_var], method="transform")

    batch = {"input": {"Test_rMOM6": {VAR_KEY: torch.randn(2, NZ, 1, H, W)}}}
    x_orig = batch["input"]["Test_rMOM6"][VAR_KEY].clone()
    out = block(batch)

    assert torch.equal(out["input"]["Test_rMOM6"][VAR_KEY], x_orig)  # untouched
    assert block._device_cache == {}
