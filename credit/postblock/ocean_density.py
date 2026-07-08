"""
ocean_density.py
----------------
OceanDensity: gen2 postblock that diagnoses seawater (potential) density from
predicted potential temperature and salinity, and optionally enforces static
stability of the density profile.

Motivation
----------
The emulator predicts ``thetao`` (potential temperature) and ``so`` (salinity)
as independent channels; nothing constrains the pair to a physically stable
water column.  Seawater density ``rho = rho(theta, S, p)`` ties them together.
This is the ocean analogue of the atmospheric mass/energy "fixer" postblocks.

There was **no seawater equation-of-state anywhere in CREDIT** before this
block (the only ``compute_density`` in the codebase is the atmospheric ideal-gas
law), so this is net-new, not a port.

Equation of state
-----------------
The default EOS is a **linear, Boussinesq, potential-density** approximation::

    rho = rho0 * (1 - alpha * (theta - theta0) + beta * (S - S0))

which is fully differentiable in pure torch (no ``gsw``, which is numpy /
non-differentiable / CPU-only and unusable inside training).  Because the
inputs are *potential* temperature, density is referenced to the surface and
pressure/compressibility are neglected.  A full TEOS-10 / Roquet-2015 75-term
polynomial can be dropped in later behind ``eos:`` — see the ``_eos_*`` methods.

Contract
--------
Operates on the reconstructed output dict, in **physical units**, so it must
run after ``Reconstruct`` and after the inverse-scaler postblock::

    batch_dict["y_processed"][source][theta_var] -> (B, n_levels, n_time, H, W)
    batch_dict["y_processed"][source][salt_var]  -> (B, n_levels, n_time, H, W)

and writes ``batch_dict["y_processed"][source][output_var]`` with the same shape.

Because ``Reconstruct`` detaches ``y_processed``, this diagnostic does not
affect the single-step training gradient; it is intended for evaluation and,
when ``enforce_stability`` is set, for shaping the rollout feed-back.

``enforce_stability`` (default ``False``)
-----------------------------------------
Enforces a statically stable profile by taking the running maximum of density
**downward** along the level dimension (density must not decrease with depth).
This adjusts the *diagnostic density field only* — it does **not** back-correct
``theta``/``S``.  It is a deliberately simple, differentiable placeholder; a
faithful convective-adjustment fixer that mixes T/S mass-weighted by layer
thickness is a documented future extension (see OCEAN_MIGRATION_NOTES.md).  It
assumes level index 0 is the surface and levels increase downward (MOM6 ``zl``
is "positive down").

Example config::

    postblocks:
      post_rollout:
        density:
          type: ocean_density
          args:
            theta_var:  "rMOM6/prognostic/3d/thetao"
            salt_var:   "rMOM6/prognostic/3d/so"
            output_var: "rMOM6/diagnostic/3d/rho"
            # optional EOS coefficients (defaults shown)
            rho0: 1027.0
            alpha: 1.7e-4
            beta:  7.6e-4
            theta0: 10.0
            s0: 35.0
            enforce_stability: false
"""

from __future__ import annotations

import torch

from credit.postblock.base import BasePostblock


class OceanDensity(BasePostblock):
    """Diagnose seawater potential density from theta and salinity.

    Args:
        theta_var: ``var_key`` of potential temperature (degC).
        salt_var: ``var_key`` of salinity (PSU).
        output_var: ``var_key`` to write density (kg/m^3) under. Defaults to the
            theta var's source/dim with name ``rho``.
        eos: EOS name. Only ``"linear"`` is implemented.
        rho0, alpha, beta, theta0, s0: linear-EOS coefficients (SI-ish units).
        enforce_stability: if True, make density non-decreasing with depth via a
            downward running max along the level dimension.
        key: which ``batch_dict`` entry holds the reconstructed dict
            (default ``"y_processed"``).
    """

    def __init__(
        self,
        theta_var: str,
        salt_var: str,
        output_var: str | None = None,
        eos: str = "linear",
        rho0: float = 1027.0,
        alpha: float = 1.7e-4,
        beta: float = 7.6e-4,
        theta0: float = 10.0,
        s0: float = 35.0,
        enforce_stability: bool = False,
        key: str = "y_processed",
    ):
        super().__init__()
        if eos != "linear":
            raise ValueError(
                f"OceanDensity: eos={eos!r} not implemented. Only 'linear' is available; "
                "extend _eos_linear with a TEOS-10 polynomial to add more."
            )
        self.theta_var = theta_var
        self.salt_var = salt_var
        self.output_var = output_var or self._default_output_key(theta_var)
        self.eos = eos
        self.rho0 = rho0
        self.alpha = alpha
        self.beta = beta
        self.theta0 = theta0
        self.s0 = s0
        self.enforce_stability = enforce_stability
        self.key = key

    @staticmethod
    def _default_output_key(theta_var: str) -> str:
        """Return ``<source>/diagnostic/<dim>/rho`` derived from the theta key."""
        parts = theta_var.split("/")
        if len(parts) == 4:
            return f"{parts[0]}/diagnostic/{parts[2]}/rho"
        return "rho"

    def _eos_linear(self, theta: torch.Tensor, salt: torch.Tensor) -> torch.Tensor:
        """Linear Boussinesq potential-density EOS."""
        return self.rho0 * (1.0 - self.alpha * (theta - self.theta0) + self.beta * (salt - self.s0))

    def forward(self, batch_dict: dict) -> dict:
        processed = batch_dict.get(self.key)
        if not isinstance(processed, dict):
            return batch_dict

        for source_vars in processed.values():
            if self.theta_var not in source_vars or self.salt_var not in source_vars:
                continue
            theta = source_vars[self.theta_var]
            salt = source_vars[self.salt_var]

            rho = self._eos_linear(theta, salt)

            if self.enforce_stability:
                # Density must not decrease with depth. Level dim is 1; index 0 is
                # the surface, increasing downward. Running max downward makes the
                # profile statically stable (adjusts density only, not theta/S).
                rho, _ = torch.cummax(rho, dim=1)

            source_vars[self.output_var] = rho

        return batch_dict
