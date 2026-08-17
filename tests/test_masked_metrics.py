"""LatWeightedMetrics with and without a validity mask.

Land cells are predicted perfectly by construction (``fill_values`` sets them to a constant 0 in
both ``y`` and ``y_pred``), so an unmasked mean counts them as free skill. These tests pin the
two properties that matter: passing no mask leaves the historical numbers untouched, and passing
one recovers the statistics of the valid subdomain.
"""

import numpy as np
import pytest
import torch

from credit.metrics import LatWeightedMetrics


def _conf(n_channels=3):
    return {
        "data": {
            "variables": [],
            "surface_variables": [f"v{i}" for i in range(n_channels)],
            "diagnostic_variables": [],
        },
        "model": {"levels": 1},
        "loss": {"use_latitude_weights": False},
        "trainer": {"ensemble_size": 1},
    }


@pytest.fixture
def case():
    torch.manual_seed(0)
    B, C, H, W = 2, 3, 8, 10
    valid = torch.zeros(1, C, H, W)
    valid[..., :4, :] = 1.0  # exactly half the grid is valid
    y = torch.randn(B, C, H, W) * valid
    pred = (0.9 * y + np.sqrt(1 - 0.81) * torch.randn(B, C, H, W)) * valid
    return LatWeightedMetrics(_conf(C)), pred, y, valid


def test_no_mask_is_unchanged(case):
    """Omitting the mask must reproduce the historical reduction exactly."""
    metrics, pred, y, _ = case
    got = metrics(pred, y)
    err = pred - y
    assert got["mae"] == pytest.approx(err.abs().mean().item(), rel=1e-5)
    assert got["mse"] == pytest.approx((err**2).mean().item(), rel=1e-5)


def test_mask_recovers_the_valid_subdomain(case):
    """With a mask, mae/mse match a direct computation over the valid cells alone."""
    metrics, pred, y, valid = case
    got = metrics(pred, y, mask=valid)
    sel = valid.expand_as(pred) > 0
    err = (pred - y)[sel]
    assert got["mae"] == pytest.approx(err.abs().mean().item(), rel=1e-5)
    assert got["mse"] == pytest.approx((err**2).mean().item(), rel=1e-5)


def test_unmasked_error_is_optimistic_by_the_valid_fraction(case):
    """The artifact itself: land dilutes every error metric by the fraction of cells it covers."""
    metrics, pred, y, valid = case
    frac = valid.mean().item()
    unmasked, masked = metrics(pred, y), metrics(pred, y, mask=valid)
    assert unmasked["mae"] < masked["mae"]
    assert masked["mae"] / unmasked["mae"] == pytest.approx(1.0 / frac, rel=0.02)
    assert masked["rmse"] / unmasked["rmse"] == pytest.approx(1.0 / np.sqrt(frac), rel=0.05)


def test_acc_is_nearly_unaffected_by_masking(case):
    """acc is a centred correlation and land sits at the field mean, so it barely moves.

    Pinned because it is the counter-intuitive half of the finding: the headline skill number
    is trustworthy even though the error numbers are not.
    """
    metrics, pred, y, valid = case
    unmasked, masked = metrics(pred, y), metrics(pred, y, mask=valid)
    assert masked["acc"] == pytest.approx(unmasked["acc"], abs=0.02)


def test_all_valid_mask_matches_no_mask(case):
    """A mask of ones is a no-op, so the two code paths agree where they overlap."""
    metrics, pred, y, _ = case
    ones = torch.ones(1, *pred.shape[1:])
    a, b = metrics(pred, y), metrics(pred, y, mask=ones)
    for k in ("acc", "rmse", "mse", "mae"):
        assert a[k] == pytest.approx(b[k], rel=1e-5), k
