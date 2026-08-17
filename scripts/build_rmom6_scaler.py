#!/usr/bin/env python
"""Build a bridgescaler JSON for the regional MOM6 ocean pipeline from pregenerated stats,
using CREDIT-style residual (xi) tendency normalization (Schreck et al. 2024, arXiv:2411.07814
Sec 3.3): T'' = (T - mu) / (xi * sigma), where sigma is the usual per-variable-per-level std
and xi (from stats_xi.nc) rescales it by that level's temporal-tendency variability so all
variables/levels contribute comparably-sized tendencies to the loss.

Converts existing per-variable mean/std/xi statistics (already computed offline, in
explore_statistics_carib12/stats/) into a bridgescaler-format scaler dict, so
bridgescaler does the actual (de)normalization at train/rollout time while the
statistics themselves come from your own precomputed files rather than a fresh
fit over the data. This follows the same construction already used by
``credit.cli._convert._build_bridgescaler_jsons`` (built for the ERA5 gen1->gen2
conversion path): a ``DStandardScalerTensor`` is populated directly —
``mean_x_``/``var_x_``/``x_columns_``/``_fit``/``n_`` — instead of calling
``.fit()`` on raw data, then serialized with ``bridgescaler.save_scaler_dict``.
``var_x_`` is set to ``(xi * std)**2`` rather than plain ``std**2``, which is the
only change from the earlier non-xi version of this script -- bridgescaler's
``DStandardScalerTensor`` has no notion of xi itself, so folding it into the
stored variance is what makes the scaler behave as xi-normalization at train time.

Inputs used (see explore_statistics_carib12/stats/):
    stats_levelwise.nc  -> <var>_mean/<var>_std per z1_l layer, for uo/vo/thetao/so
    stats_global.nc     -> SSH_mean/SSH_std (single scalar; SSH has no level dim)
    stats_xi.nc         -> <var>_xi per z1_l layer (uo/vo/thetao/so) / scalar (SSH)

The 5 prognostic variables (uo, vo, thetao, so, SSH) get input- *and* target-side scaler
entries. The 6 input-only channels — dynamic_forcing (taux, tauy, net_heat_surface, runoff)
and static (deptho, wet) — get **input-side entries only**, read from
``stats_global_surface.nc`` (built by ``scripts/build_rmom6_surface_stats.py``); they are
never predicted, so a target-side entry would be meaningless and the postblock, which reads
``["target"]``, correctly never sees them.

Leaving those 6 unscaled is not benign: measured on the input tensor, ``deptho`` carries
std ~2276 (max 6000) against the prognostics' ~1, and ``net_heat_surface`` ~108, while
``runoff`` (~2e-4) is effectively invisible. Since WxFormer uses ``patch_height/width: 1``,
its cross-embed conv sees those magnitudes directly and the input is dominated by static
bathymetry — which shows up in rollouts as a bathymetry-shaped imprint on the surface fields.

``preblocks.normalize.args.variables`` should be the explicit 11-variable list this script
prints, not ``[]`` (empty = "every variable present"): if any variable in the batch lacks a
matching scaler entry, bridgescaler's ``scale_var_dict`` raises ``AssertionError`` on the
first unscaled key it walks into ("...found in 'var_dict' but missing in 'scalers'").

Output is a single JSON, structured ``{"input": {source: {var_key: scaler}}, "target":
{source: {var_key: scaler}}}``. Point *both* ``preblocks.per_step.normalize.args.scaler_path``
and ``postblocks.per_step.denorm.args.scaler_path`` at it — this mirrors the existing
gen2 ERA5 example configs, where both blocks share the same "_pre.json" file (the
postblock's own loader unwraps ``["target"]`` from it internally; see
``credit/postblock/scaler.py``).

If ``--levels`` is given, it must exactly match the ``--levels`` used for
``preprocess_rmom6.py`` — the scaler's per-level vector (and xi) is nearest-matched
against ``stats_levelwise.nc``/``stats_xi.nc``'s shared ``z1_l`` coordinate the same
way, so the channel order/count lines up with the preprocessed data.

``--level-pairs`` is the counterpart for a ``preprocess_rmom6.py --level-pairs`` run (25
thickness-weighted-merged levels instead of a sparse native-level subset). Nearest-matching
doesn't apply here -- a merged level's mean/std/xi were computed directly on the merged data by
``scripts/build_rmom6_levelpair_stats.py`` (see that script's docstring for why they can't be
derived from the native-level stats: std/xi depend on the cross-level covariance between the
two merged native layers, which the native-level stats files don't record), so this just reads
``stats_{levelwise,global,xi}_levelpairs.nc`` and uses their 25 levels as-is, no ``.sel()``.
Mutually exclusive with ``--levels``.

Usage::

    # All 50 levels (matches the preprocess_rmom6.py default)
    python scripts/build_rmom6_scaler.py

    # Matches a preprocess_rmom6.py run done with --levels 0.5,5,15,30,60,100
    python scripts/build_rmom6_scaler.py --levels 0.5,5,15,30,60,100

    # Matches a preprocess_rmom6.py --level-pairs run (needs stats_*_levelpairs.nc built by
    # build_rmom6_levelpair_stats.py first)
    python scripts/build_rmom6_scaler.py --level-pairs

Dependencies: xarray, numpy, torch, bridgescaler.
"""

from __future__ import annotations

import argparse
import os
from getpass import getuser

import numpy as np
import torch
import xarray as xr
from bridgescaler import save_scaler_dict
from bridgescaler.distributed_tensor import DStandardScalerTensor

DEFAULT_STATS_DIR = "/glade/work/ajanney/RegionalEmulation_v2/explore_statistics_carib12/stats"
SOURCE_NAME = "rMOM6"
LEVEL_VARS = ["thetao", "so", "uo", "vo"]  # covered by stats_levelwise.nc
GLOBAL_VARS = ["SSH"]  # covered by stats_global.nc (scalar, no level dim)
# Input-only 2D channels, covered by stats_global_surface.nc (build_rmom6_surface_stats.py).
# Level-independent, so the same file serves both the native-level and --level-pairs paths.
# xi is a tendency scaling for predicted fields only, so it is never applied to these.
SURFACE_VARS: dict[str, list[str]] = {
    "dynamic_forcing": ["taux", "tauy", "net_heat_surface", "runoff"],
    "static": ["deptho", "wet"],
}


def parse_levels(spec: str | None) -> list[float] | None:
    if spec is None:
        return None
    return [float(v) for v in spec.split(",")]


def _make_scaler(mean_vals: np.ndarray, std_vals: np.ndarray) -> DStandardScalerTensor:
    mean_vals = np.atleast_1d(np.asarray(mean_vals, dtype=np.float32))
    std_vals = np.atleast_1d(np.asarray(std_vals, dtype=np.float32))
    sc = DStandardScalerTensor(channels_last=False)
    sc.mean_x_ = torch.tensor(mean_vals, dtype=torch.float32)
    sc.var_x_ = torch.tensor(std_vals**2, dtype=torch.float32)
    sc.x_columns_ = list(range(len(mean_vals)))
    sc._fit = True
    sc.n_ = 1
    return sc


def build_scaler_dict(
    stats_dir: str, levels: list[float] | None, use_xi: bool = True, level_pairs: bool = False
) -> tuple[dict, list[str]]:
    """Return (pre_dict, prognostic_var_keys) built from the stats.nc files.

    pre_dict has the structure ``{"input": {SOURCE_NAME: {var_key: scaler}},
    "target": {SOURCE_NAME: {var_key: scaler}}}`` expected by
    ``credit.preblock.scaler.BridgeScalerTransform``; the postblock reads the
    same file and extracts just ``["target"]`` (see module docstring).

    Effective std stored in each scaler is ``xi * std`` (xi tendency normalization,
    see module docstring) when ``use_xi`` is True (the default); with ``use_xi=False``
    the raw ``std`` from stats_levelwise.nc/stats_global.nc is used unscaled -- this is the
    "tendency normalization off" arm of the normalization x rollout-length x tendency
    experiment grid, see ``EXPERIMENT_DESIGN.md``.

    ``level_pairs`` reads the ``_levelpairs`` stats files (built by
    ``build_rmom6_levelpair_stats.py`` directly from level-paired preprocessed data) and skips
    the nearest-match ``.sel()`` -- their 25 merged levels are used exactly as stored, since
    nearest-matching a sparse subset doesn't apply to already-merged levels. Mutually exclusive
    with ``levels`` (enforced in ``main()``).
    """
    suffix = "_levelpairs" if level_pairs else ""
    levelwise = xr.open_dataset(os.path.join(stats_dir, f"stats_levelwise{suffix}.nc"))
    globalwise = xr.open_dataset(os.path.join(stats_dir, f"stats_global{suffix}.nc"))
    xi_ds = xr.open_dataset(os.path.join(stats_dir, f"stats_xi{suffix}.nc")) if use_xi else None

    pre_input: dict[str, DStandardScalerTensor] = {}
    pre_target: dict[str, DStandardScalerTensor] = {}
    var_keys: list[str] = []

    for varname in LEVEL_VARS:
        mean_da = levelwise[f"{varname}_mean"]
        std_da = levelwise[f"{varname}_std"]
        if levels is not None:
            mean_da = mean_da.sel(z1_l=levels, method="nearest")
            std_da = std_da.sel(z1_l=levels, method="nearest")
        if use_xi:
            xi_da = xi_ds[f"{varname}_xi"]
            if levels is not None:
                xi_da = xi_da.sel(z1_l=levels, method="nearest")
            eff_std = std_da.values * xi_da.values
        else:
            eff_std = std_da.values
        sc = _make_scaler(mean_da.values, eff_std)
        key = f"{SOURCE_NAME}/prognostic/3d/{varname}"
        pre_input[key] = sc
        pre_target[key] = sc
        var_keys.append(key)

    for varname in GLOBAL_VARS:
        mean_val = float(globalwise[f"{varname}_mean"].values)
        std_val = float(globalwise[f"{varname}_std"].values)
        eff_std = std_val * float(xi_ds[f"{varname}_xi"].values) if use_xi else std_val
        sc = _make_scaler(mean_val, eff_std)
        key = f"{SOURCE_NAME}/prognostic/2d/{varname}"
        pre_input[key] = sc
        pre_target[key] = sc
        var_keys.append(key)

    # Input-only channels: added to pre_input ONLY. The postblock reads ["target"], and these
    # are never predicted, so a target-side entry would be meaningless. Without them the
    # channels reach the model unscaled -- deptho at ~2300x the prognostics' scale, which
    # imprints bathymetry on the predicted surface fields.
    surface_path = os.path.join(stats_dir, "stats_global_surface.nc")
    surface_keys: list[str] = []
    if os.path.exists(surface_path):
        surface = xr.open_dataset(surface_path)
        for field_type, varnames in SURFACE_VARS.items():
            for varname in varnames:
                if f"{varname}_mean" not in surface:
                    print(f"WARNING: {varname} missing from {surface_path}; skipping")
                    continue
                key = f"{SOURCE_NAME}/{field_type}/2d/{varname}"
                pre_input[key] = _make_scaler(
                    float(surface[f"{varname}_mean"].values), float(surface[f"{varname}_std"].values)
                )
                surface_keys.append(key)
        surface.close()
    else:
        print(f"WARNING: {surface_path} missing -- run scripts/build_rmom6_surface_stats.py;")
        print("         dynamic_forcing/static channels will reach the model UNSCALED.")

    levelwise.close()
    globalwise.close()
    if xi_ds is not None:
        xi_ds.close()

    pre_dict = {"input": {SOURCE_NAME: pre_input}, "target": {SOURCE_NAME: pre_target}}
    return pre_dict, var_keys + surface_keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stats-dir", default=DEFAULT_STATS_DIR, help="Directory with stats_levelwise.nc / stats_global.nc."
    )
    parser.add_argument(
        "--out",
        default=f"/glade/derecho/scratch/{getuser()}/rmom6_regional/scaler/ocean_bridgescaler_levelwise_xi.json",
        help="Output path for the combined scaler JSON.",
    )
    parser.add_argument(
        "--levels",
        default=None,
        help="Comma-separated target depths in metres, nearest-matched to z1_l. "
        "Must match the --levels used for preprocess_rmom6.py. Omit for all 50 levels. "
        "Mutually exclusive with --level-pairs.",
    )
    parser.add_argument(
        "--level-pairs",
        action="store_true",
        help="Read stats_{levelwise,global,xi}_levelpairs.nc (built by "
        "build_rmom6_levelpair_stats.py) instead, matching a preprocess_rmom6.py --level-pairs "
        "run's 25 thickness-merged levels. No nearest-match subsetting. Mutually exclusive with "
        "--levels.",
    )
    parser.add_argument(
        "--no-xi",
        action="store_true",
        help="Skip xi tendency scaling -- std stays exactly as stats_levelwise.nc/stats_global.nc "
        "report it, with no per-level tendency-variance rescaling. Default: xi applied.",
    )
    args = parser.parse_args()

    if args.level_pairs and args.levels:
        parser.error("--levels and --level-pairs are mutually exclusive")
    levels = parse_levels(args.levels)
    pre_dict, var_keys = build_scaler_dict(args.stats_dir, levels, use_xi=not args.no_xi, level_pairs=args.level_pairs)

    out = args.out
    if args.level_pairs and out == parser.get_default("out"):
        out = out.replace("ocean_bridgescaler_levelwise_xi.json", "ocean_bridgescaler_levelwise_xi_levelpairs.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_scaler_dict(pre_dict, out)

    print(f"stats dir : {args.stats_dir}")
    print(f"tendency  : {'off (--no-xi)' if args.no_xi else 'on (xi-scaled)'}")
    print(
        f"levels    : {'pairwise thickness-average (50 -> 25)' if args.level_pairs else ('all 50' if levels is None else levels)}"
    )
    print(f"wrote     : {out}")
    predicted = [k for k in var_keys if "/prognostic/" in k]
    input_only = [k for k in var_keys if "/prognostic/" not in k]

    print()
    print("preblocks.normalize.args.variables — ALL covered variables (everything fed to the model):")
    for key in var_keys:
        print(f"  - {key}")
    print()
    print("postblocks.denorm.args.variables — PROGNOSTICS ONLY (the input-only channels below are")
    print("never predicted, so there is nothing to invert for them):")
    for key in predicted:
        print(f"  - {key}")
    print()
    if input_only:
        print(
            f"Input-only channels now normalized ({len(input_only)}): "
            + ", ".join(k.split("/")[-1] for k in input_only)
        )
        print("Leaving these out of the preblock list makes them enter the model RAW — deptho at ~2300x")
        print("the prognostics' scale — which imprints bathymetry on the predicted surface fields.")


if __name__ == "__main__":
    main()
