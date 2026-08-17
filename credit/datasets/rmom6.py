"""
rmom6.py
--------
RegionalMOM6Dataset: gen2 dataset for processed Regional MOM6 ocean output.

This is a thin subclass of :class:`credit.datasets.local.LocalDataset`.  The
regional-ocean CREDIT input files are ordinary NetCDF/Zarr with dims
``(time, zl, yh, xh)`` for 3D fields and ``(time, yh, xh)`` for 2D fields, so
``LocalDataset``'s generic ``_extract_field`` already does almost everything:
per-variable tensors, level selection via ``level_coord``, cftime handling,
and the nested ``{source}/{field_type}/{dim}/{varname}`` output schema consumed
by the preblock/postblock pipeline.

Two ocean-specific wrinkles are handled here and are the *only* reason this
subclass exists:

1. **Static file carries a singleton ``time`` dimension.**  MOM6 writes the
   static/geometry file with a length-1 ``time`` axis at the model epoch
   (e.g. ``0001-01-01``).  ``LocalDataset`` would try ``ds.sel(time=t)`` with
   the *training* timestamp and raise ``KeyError``.  For ``static`` fields we
   instead take ``isel(time=0)`` — static data is time-invariant by definition.

2. **File variable names differ from the canonical CREDIT names.**  MOM6 stores
   potential temperature as ``temp`` and salinity as ``salt``; downstream ocean
   postblocks (density, clamp) are easiest to configure against canonical names
   (``thetao``, ``so``).  An optional ``rename`` map in the source config maps
   ``canonical_name -> file_name`` so the config can list canonical names while
   the loader reads the file-native variable.

Open boundary conditions (the ``north``/``east`` nudging slices) are **not**
handled by this dataset.  They live on a different vertical grid
(``z_boundary`` vs ``zl``) and a different calendar, and the migration plan
folds them into ``dynamic_forcing`` via an offline preprocessing step (see
OCEAN_MIGRATION_NOTES.md).  Once baked to full-domain forcing fields they are
loaded like any other ``dynamic_forcing`` variable — no code here required.

Example YAML::

    data:
      source:
        rMOM6:
          dataset_type: "rmom6"
          level_coord: "zl"
          levels: [ ... ]                 # subset of zl layers, or omit for all
          rename:                          # optional canonical -> file name
            thetao: temp
            so: salt
          variables:
            prognostic:
              vars_3D: [ 'uo', 'vo', 'thetao', 'so' ]
              vars_2D: [ 'SSH' ]
              path: '/.../CREDIT_Input.*.credit_ocean_state.nc'
            dynamic_forcing:
              vars_2D: [ 'taux', 'tauy', 'net_heat_surface', 'p_surf' ]
              path: '/.../CREDIT_Input.*.credit_fluxes.nc'
            static:
              vars_2D: [ 'deptho', 'wet' ]
              path: '/.../CREDIT_Input.*.static.nc'
      start_datetime: "2000-01-01"
      end_datetime:   "2000-12-31"
      timestep: "1d"
      forecast_len: 1
"""

from __future__ import annotations

from typing import Any

import cftime
import pandas as pd
import torch
import xarray as xr

from credit.datasets._utils import _find_file, _to_cftime  # pyright: ignore[reportPrivateUsage]
from credit.datasets.local import LocalDataset


class RegionalMOM6Dataset(LocalDataset):
    """PyTorch Dataset for processed Regional MOM6 ocean data (gen2 schema).

    See module docstring for the differences from ``LocalDataset`` and an
    example config.
    """

    def __init__(self, data_config: dict[str, Any], return_target: bool = False) -> None:
        # LocalDataset.__init__ asserts dataset_type == "local"; bypass that by
        # normalising the assertion target, then restore our own type string.
        super(LocalDataset, self).__init__(data_config, return_target)

        self.dataset_type = "rmom6"
        self.level_coord: str | None = self.curr_source_cfg.get("level_coord", "zl")
        self.levels: list | None = self.curr_source_cfg.get("levels")
        # canonical_name -> file_name (e.g. {"thetao": "temp", "so": "salt"})
        self.rename: dict[str, str] = self.curr_source_cfg.get("rename", {}) or {}
        self.static_metadata: dict[str, Any] = {
            "levels": self.levels,
            "datetime_fmt": "unix_ns",
        }
        self.mode = "local"
        self.time_coord = self.curr_source_cfg.get("time_coord", "time")
        # path -> open xr.Dataset, populated lazily inside __getitem__ (i.e. per DataLoader
        # worker process, after fork/spawn -- never touched here in __init__, so this stays
        # an empty, trivially picklable dict at worker-spawn time). See _open_cached.
        self._open_cache: dict[str, xr.Dataset] = {}
        self.init_register_all_fields()

    def _extract_field(
        self,
        field_type: str,
        t: pd.Timestamp,
        sample: dict[str, Any],
    ) -> None:
        """Open the file for *field_type* at time *t* and populate *sample*.

        Mirrors ``LocalDataset._extract_field`` with two ocean-specific changes:
        static fields are read with ``isel(time=0)`` (singleton MOM6 time axis),
        and variable names are resolved through ``self.rename`` so config may use
        canonical names.
        """
        file_intervals = self.file_dict.get(field_type)
        if not file_intervals or field_type not in self.var_dict:
            return

        vd = self.var_dict[field_type]
        vars_3D: list[str] = vd["vars_3D"]
        vars_2D: list[str] = vd["vars_2D"]

        ds = self._open_cached(_find_file(file_intervals, t))
        ds_t = self._select_time(ds, field_type, t)

        # 3D variables: (n_levels, lat, lon) -> (n_levels, 1, lat, lon)
        for vname in vars_3D:
            file_var = self.rename.get(vname, vname)
            da = ds_t[file_var]
            if self.levels is None:
                arr = da.values
                if self.level_coord in ds_t.coords:
                    self.levels = ds_t[self.level_coord].values.tolist()
                    self.static_metadata["levels"] = self.levels
            else:
                arr = da.sel({self.level_coord: self.levels}, method="nearest").values
            tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(1)
            sample[self._get_field_name(field_type, "3d", vname)] = tensor

        # 2D variables: (lat, lon) -> (1, 1, lat, lon)
        for vname in vars_2D:
            file_var = self.rename.get(vname, vname)
            arr = ds_t[file_var].values
            tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            sample[self._get_field_name(field_type, "2d", vname)] = tensor

    def _open_cached(self, path: str) -> xr.Dataset:
        """Open *path* once per worker process and reuse the handle for every later sample.

        Previously reopened (and closed) a fresh ``xr.Dataset`` on every single call -- cheap
        in wall-clock time (consolidated Zarr metadata makes an open ~0.011s), but each open
        allocates a full set of index/metadata Python objects that a same-process reopen a few
        milliseconds later doesn't need. Under the real training DDP layout (4 ranks/node x
        (train+valid) thread_workers each), many workers touching a never-before-opened
        store (e.g. crossing a year boundary) at the same moment produces simultaneous
        transient RSS spikes per worker -- confirmed empirically (a diagnostic run tracking
        each persistent DataLoader worker's RSS across 3 epochs showed non-monotonic
        per-worker spikes up to ~9x the steady-state size, never a cumulative/monotonic leak).
        Caching the open handle (keyed by path, populated lazily inside __getitem__ so it's
        per-worker-process state, not touched at __init__/pickle time) means each store's
        consolidated metadata is parsed once per worker instead of on every sample.

        The cache only ever holds the small, fixed number of distinct store paths a training
        run touches (one zarr per field type per year, not per sample), so it stays bounded
        regardless of dataset length.
        """
        cached = self._open_cache.get(path)
        if cached is not None:
            return cached
        if str(path).endswith(".zarr"):
            ds = xr.open_dataset(path, engine="zarr", consolidated=True)
        else:
            ds = xr.open_dataset(path)
        self._open_cache[path] = ds
        return ds

    def _select_time(self, ds: xr.Dataset, field_type: str, t: pd.Timestamp) -> xr.Dataset:
        """Select the timestep from *ds*, handling MOM6's singleton static time axis."""
        if self.time_coord not in ds.dims:
            return ds

        # Static fields are time-invariant; MOM6 stores them with a length-1 time
        # axis at the model epoch, which will not match a training timestamp.
        if field_type == "static":
            return ds.isel({self.time_coord: 0})

        if isinstance(ds[self.time_coord].values[0], cftime.datetime):
            calendar = ds[self.time_coord].values[0].calendar
            t_sel = _to_cftime(t, calendar)
        else:
            t_sel = t
        return ds.sel({self.time_coord: t_sel})
