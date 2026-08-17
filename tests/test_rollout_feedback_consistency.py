"""Rollout feedback must re-enter the model on the same footing as the dataset's own input.

This guards the bug class that invalidated every rMOM6 multistep run: a postblock wrote a
*physical-space* constant into masked cells, so after ``normalize`` those cells arrived at step 2
as ``(0 - mu)/sigma`` -- anywhere from -10 to -3e5 -- where step 1 had a clean normalized 0.
Nothing raised. The loss fell three orders of magnitude and the model was uncorrelated with truth.

The invariant these tests pin down is cheap to state and cheap to check: **whatever a masking
postblock writes into a cell, that cell must land on the same normalized value as the dataset's
own missing-data convention produces.** Any postblock that blanks cells in physical space and
feeds them back through a normalizing preblock is subject to it.
"""

import pytest
import torch

from credit.preblock.fill_values import FillValues


FILL_RULES = [{"search": "nan", "fill": 0.0}]


def _normalize(x, mu, sigma):
    return (x - mu) / sigma


def _dataset_step1(raw, mu, sigma, land):
    """What the dataset hands the model at t=0: land is NaN, normalize, then fill_values."""
    x = raw.clone()
    x[land] = float("nan")
    x = _normalize(x, mu, sigma)
    batch = {"input": {"src": {"src/v": x}}}
    return FillValues(rules=FILL_RULES, variables=["src/v"], data_types=["input"])(batch)["input"]["src"]["src/v"]


def _rollout_step2(pred, mu, sigma, land, blank_with_nan):
    """What assemble_rollout_batch hands the model at t=1: the postblock blanks masked cells in
    physical space, then the same normalize -> fill_values chain runs again."""
    y = pred.clone()
    y[land] = float("nan") if blank_with_nan else 0.0
    y = _normalize(y, mu, sigma)
    batch = {"input": {"src": {"src/v": y}}}
    return FillValues(rules=FILL_RULES, variables=["src/v"], data_types=["input"])(batch)["input"]["src"]["src/v"]


@pytest.fixture
def field():
    torch.manual_seed(0)
    raw = torch.rand(1, 4, 1, 6, 6) * 5.0 + 20.0  # physical units, e.g. degC
    land = torch.zeros_like(raw, dtype=torch.bool)
    land[..., :3, :] = True  # half the domain
    mu = torch.tensor(26.0)
    sigma = torch.tensor(1.4)
    return raw, land, mu, sigma


def test_nan_blanking_matches_the_dataset_convention(field):
    """The fix: blanking with NaN puts step 2's masked cells exactly where step 1's are."""
    raw, land, mu, sigma = field
    step1 = _dataset_step1(raw, mu, sigma, land)
    step2 = _rollout_step2(raw, mu, sigma, land, blank_with_nan=True)
    assert torch.equal(step1[land], step2[land])
    assert torch.all(step2[land] == 0.0)


def test_zero_blanking_is_a_distribution_shift(field):
    """The bug: blanking with physical 0 sends masked cells far from where step 1 put them.

    Pinned as a test so the regression is visible rather than silent -- this is the behaviour
    ``masked_fill_nan: False`` still selects, and it must never be the default again.
    """
    raw, land, mu, sigma = field
    step1 = _dataset_step1(raw, mu, sigma, land)
    step2 = _rollout_step2(raw, mu, sigma, land, blank_with_nan=False)
    assert not torch.allclose(step1[land], step2[land])
    # (0 - 26)/1.4 = -18.6, against a step-1 value of exactly 0.
    assert step2[land].max() < -18.0


def test_masked_cells_do_not_shift_the_field_statistics(field):
    """Whatever lands in the masked cells must not move the *unmasked* cells' distribution.

    A cheap proxy for "step 2 looks like step 1 to the network": the ocean cells are identical
    and the masked cells contribute the same constant, so mean and std over the full grid match.
    """
    raw, land, mu, sigma = field
    step1 = _dataset_step1(raw, mu, sigma, land)
    step2 = _rollout_step2(raw, mu, sigma, land, blank_with_nan=True)
    assert torch.allclose(step1[~land], step2[~land])
    assert step1.mean().item() == pytest.approx(step2.mean().item(), abs=1e-6)
    assert step1.std().item() == pytest.approx(step2.std().item(), abs=1e-6)


@pytest.mark.parametrize("mu,sigma", [(26.0, 1.4), (35.6, 15.3), (4.0, 1.1e-4)])
def test_invariant_holds_for_any_scaler(field, mu, sigma):
    """The NaN convention is scaler-independent; the zeroing bug's severity is not.

    sigma=1.1e-4 is the order of the xi-scaled deep-salinity channels, where zeroing produced
    the -3e5 values. NaN blanking is unaffected by how extreme the scaler is.
    """
    raw, land, _, _ = field
    mu, sigma = torch.tensor(mu), torch.tensor(sigma)
    step1 = _dataset_step1(raw, mu, sigma, land)
    step2 = _rollout_step2(raw, mu, sigma, land, blank_with_nan=True)
    assert torch.equal(step1[land], step2[land])


def test_ocean_wet_mask_defaults_to_nan_blanking():
    """The real postblock's default must stay NaN, not 0.

    The tests above model the two behaviours by hand; this one pins the actual default, so
    flipping ``masked_fill_nan`` back to False -- which is what every invalid multistep run was
    trained under -- fails here instead of silently three orders of magnitude into a loss curve.
    """
    import inspect

    from credit.postblock.ocean_wet_mask import OceanWetMask

    default = inspect.signature(OceanWetMask.__init__).parameters["masked_fill_nan"].default
    assert default is True, (
        "OceanWetMask.masked_fill_nan defaulted to False; physical-space zeroing corrupts the "
        "multi-step feedback path (see the module docstring)."
    )


def test_numeric_fill_rules_never_swallow_nan():
    """``fill_values`` must not let a numeric rule capture NaN.

    If ``search: 0.0`` matched NaN, the land convention would silently depend on rule order and
    the invariant above would hold only by accident.
    """
    x = torch.tensor([[[[[float("nan"), 0.0, 1.0]]]]])
    rules = [{"search": 0.0, "op": "==", "fill": 99.0}]
    out = FillValues(rules=rules, variables=["src/v"], data_types=["input"])({"input": {"src": {"src/v": x}}})["input"][
        "src"
    ]["src/v"]
    assert torch.isnan(out[..., 0]), "a numeric rule matched a NaN cell"
    assert out[..., 1] == 99.0
    assert out[..., 2] == 1.0
