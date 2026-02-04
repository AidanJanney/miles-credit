import xarray as xr
import pandas as pd
import numpy as np
import cftime
import torch
from torch.utils.data import Dataset
import logging
from glob import glob

logger = logging.getLogger(__name__)
    
VALID_FIELD_TYPES = {"prognostic", "dynamic_forcing", "static", "diagnostic", "east_boundary", "south_boundary", "west_boundary", "north_boundary"}

class RegionalMOM6Dataset(Dataset):
    """ Pytorch Dataset for processed Regional MOM6 data. Relies on a configuration dictionary to define:
            1) 2D / 3D variables
            2) Start, End and Frequency of Datetimes
            3) the base path to the directory where the data is stored. 
            4) Example YAML Format:

                data:
                    source:
                    regional_MOM6:
                        prognostic:
                            # 
                            vars_3D: ['uo', 'vo', 'thetao', 'so']
                            vars_2D: ['SSH']
                            path: '/glade/derecho/scratch/ajanney/Regional_Ocean_Emulation/CESM_Data_Preprocessing/carib12_runoff_tides_rmax600_f200_gioGlofas_gioNNSM/*full_fields*.zarr'
                            transform:
                                mean_path: '/glade/derecho/scratch/ajanney/Regional_Ocean_Emulation/CESM_Data_Preprocessing/carib12_runoff_tides_rmax600_f200_gioGlofas_gioNNSM/statistics/carib12_emulation_prognostic_mean.nc'
                                std_path: '/glade/derecho/scratch/ajanney/Regional_Ocean_Emulation/CESM_Data_Preprocessing/carib12_runoff_tides_rmax600_f200_gioGlofas_gioNNSM/statistics/carib12_emulation_prognostic_std.nc'
                            
                        ...
                
                    start_datetime: "2000-01-01" 
                    end_datetime: "2000-12-31"
                    timestep: "1d"
        
        Assumptions:
            1) The data must be stored in yearly zarr files with a unique 4-digit year (YYYY) in the file name
            2) "time" dimension / coordinate is present with the datetime64[ns] datatype
            3) "level" dimension name representing the vertical level
            4) Dimention order of ('time', level', 'latitude', 'longitude') for 3D vars (remove level for 2D)
            5) Stored Zarr data should be chunked efficiently for a fast read (recommend small chunks across time dimension).
            
    """
    
    def __init__(self, config, return_target=False):
        self.source_name = "regional_MOM6"
        self.return_target = return_target
        self.dt = pd.Timedelta(config["timestep"])
        self.num_forecast_steps = config["forecast_len"] + 1
        self.start_datetime = pd.Timestamp(config["start_datetime"])
        self.end_datetime = pd.Timestamp(config["end_datetime"])
        self.datetimes = self._timestamps()
        self.years = [str(y) for y in self.datetimes.year]
        self.file_dict = {}
        self.var_dict = {}

        for field_type, d in config["source"][self.source_name].items():
            if field_type not in VALID_FIELD_TYPES:
                raise KeyError(
                    f"Unknown field_type '{field_type}' in config['source']['{self.source_name}']. "
                    f"Valid options are: {sorted(VALID_FIELD_TYPES)}"
                )

            if isinstance(d, dict):
                if not d.get("vars_3D") and not d.get("vars_2D"):
                    raise ValueError(
                        f"Field '{field_type}' must define at least one of vars_3D or vars_2D"
                    )

                files = sorted(glob(d.get("path", "")))
                self.file_dict[field_type] = self._map_files(files) if files else None
                self.var_dict[field_type] = {
                    "vars_3D": d.get("vars_3D", []),
                    "vars_2D": d.get("vars_2D", []),
                }
            else:
                self.file_dict[field_type] = None
                
    def _timestamps(self):
        """
        return total time steps
        """
        return pd.date_range(
            self.start_datetime,
            self.end_datetime - self.num_forecast_steps * self.dt,
            freq=self.dt,
        )
        
    def __len__(self):
        # total number of valid start times
        return len(self.init_times)
    
    def _map_files(self, file_list):
        """
        Create a dictionary to lookup the file for a timestep

        Args:
             file_list (list): List of file paths
        """
        if len(file_list) > 1: 
        # unique years string (note needs to be separate years in filename)
            file_map = {int(y): f for f in file_list for y in self.years if y in f}
        else:
            file_map = {int(y): file_list[0] for y in self.years}

        return file_map
    
    def __getitem__(self, args):
        """
        Returns a sample of data.

        Args:
            args (tuple): Input_time step from sampler, step index from sampler
        """
        return_data = {"metadata": {}}
        t, i = args
        t = pd.Timestamp(t)
        t_target = t + self.dt

        # always load dynamic forcing
        self._open_ds_extract_fields("dynamic_forcing", t, return_data)

        # load prognostic and static if first time step
        if i == 0:
            self._open_ds_extract_fields("static", t, return_data)
            self._open_ds_extract_fields("prognostic", t, return_data)

        # load t+1 if training
        if self.return_target:
            for key in ("prognostic", "diagnostic"):
                if key in self.file_dict.keys():
                    self._open_ds_extract_fields(
                        key, t_target, return_data, is_target=True
                    )
            self._pop_and_merge_targets(return_data)

        self._add_metadata(return_data, t, t_target)

        return return_data

    def _rollout_mode(self, rollout_init_times, time_tol):
        
        logger.info(f"selecting rollout times with time tolerance {time_tol}")
        self.init_times = self.init_times.sel(t=rollout_init_times, method="nearest",
                                              tolerance=pd.Timedelta(time_tol[0], time_tol[1])
                                              )

    def _generate_valid_init_times(self, valid_init_filepath):
        # due to missing data, need to have a different list of valid init times
        def check_valid_forecast_times(t_init, timestep, num_forecast_steps):
            time_tolerance = pd.Timedelta(11, "m")
            target_times = [t_init.values + timestep * step for step in range(1, num_forecast_steps + 1)]
            zarr_times = self.ds.t.sel(t=target_times, method="nearest")
            within_tol = (zarr_times.values - np.array(target_times).astype(zarr_times.dtype)) < time_tolerance

            return all(within_tol)

        logger.info(f"generating valid init times and saving to {valid_init_filepath}. will take around 5 min")

        valid_times = []
        for t in self.ds.t:
            if check_valid_forecast_times(t, self.timestep, self.num_forecast_steps):
                valid_times.append(t.values)

        valid_times_da = xr.DataArray(valid_times, coords={"t": valid_times})

        times_to_drop = xr.open_dataarray(join(self.valid_init_dir, "nan_times.nc"))
        valid_times_da = valid_times_da[ ~ valid_times_da.t.isin(times_to_drop)]

        valid_times_da.to_netcdf(valid_init_filepath)
        logger.info(f"wrote valid init times to {valid_init_filepath}")

        return valid_times_da
    
    def inverse_transform_ABI(self, da):
        # da must be in the same order as the source data
        # channels must be the first dimension
        unscaled = da * self.scaler_ds["std"] + self.scaler_ds["mean"]

        if self.log_normal_scaling:
            da[0] = np.exp(unscaled[0])

        return da
        
    def _normalize_ABI(self, da):
        return (da - self.scaler_ds["mean"]) / self.scaler_ds["std"]

    def _nanfill_ABI(self, da):
        return da.fillna(0.0)

    def __getitem__(self, args):
        # default: load target state
        ts, mode = args

        ds = self.ds.sel(t=ts, method="nearest")
        # no need to check time tolerance, should be taken care of by init time generation
        
        time_str = pd.Timestamp(ds.t.values).strftime("%Y-%m-%dT%H:%M:%S")

        return_data = {"mode": mode,
                    "stop_forecast": mode == "stop",
                    "datetime": time_str,}
        
        if mode != "forcing": # draw goes if mode is not forcing
            da = ds["BT_or_R"].copy()

            if self.log_normal_scaling: #channels is the first axis
                da[0] = np.log(da[0])
            
            da = self._normalize_ABI(da)
            da = self._nanfill_ABI(da)
            
            # da.shape = c, 1003, 923
            data = torch.tensor(da.values).unsqueeze(1)
            # data = torch.nn.functional.pad(data, (0,0,11,10), "replicate")
            if self.padding:
                data = torch.nn.functional.pad(data, (19,18,11,10), "constant", 0.0)
            # coerce to c, t, 1024, 960 to work with wxformer
        
        if self.era5dataset and mode != "stop": # always draw era5 if not stopping
            return_data["era5"] = self.get_era5(ds)

        if mode == "init":
            return_data["x"] = data
            return return_data
        elif mode == "forcing":
            return return_data
        elif mode == "y" or mode == "stop":
            return_data["y"] = data
            return return_data
        else:
            raise ValueError(f"{mode} is not a valid sampling mode")
        
    def get_era5(self, ds):
        # TODO: how to sample era5 forcing? next hour?

        ts = pd.Timestamp(ds.t.values)
        era5_ts = ts.round("h") # round to the nearest hour
        
        # interpolated data source baked into paths we init era5 with
        era5_data = self.era5dataset[(era5_ts, "init")]
        era5_data["timedelta_seconds"] = int((era5_ts - ts).total_seconds())

        return era5_data

# class ERA5Interpolator:
#     def __init__(self,
#                  data_conf: Dict,
#                  era5dataset: Dataset = None,):
#         """
#         taking advantage of DistributedSampler class code with this dataset
        
#         Args:
#             ds: xr dataset with a time attribute

#             example time config:
#             {"timestep": pd.Timedelta(1, "h"),
#                 "num_forecast_steps": 1
#                  },
#         """
#         self.era5dataset = era5dataset
#         # setup regridder
#         self.regridder = None
#         if era5dataset and data_conf.get("regrid_loc", None):
#             regrid_loc = data_conf["regrid_loc"]
#             da_outgrid = xr.open_dataset(data_conf["outgrid_loc"], engine="h5netcdf")
#             ds_ingrid = xr.open_dataset(data_conf["ingrid_loc"], engine="h5netcdf")
#             self.regridder = xe.Regridder(ds_ingrid, da_outgrid, 'bilinear', unmapped_to_nan=True, weights=regrid_loc)

#     def __getitem__(self, args):
#         # default: load target state
#         ts, mode = args

#         # run interpolation
#         era5_ds_dict = self.era5dataset[(ts, "y_xarray")] #draw an xarray
#         field_types = ["prognostic", "dynamic_forcing"]
#         combined_ds = xr.merge([era5_ds_dict[field] for field in field_types])
#         regridded = self.regridder(combined_ds, skipna=True, na_thres=1.0)

#         return regridded
#         # ts = pd.Timestamp(combined_ds.time.values)
#         # save_dir = os.path.join("/glade/derecho/scratch/dkimpara/goes-cloud-dataset/era5_regrid/",
#         #                          str(ts.year))
#         # os.makedirs(save_dir, exist_ok=True)
#         # regridded.to_netcdf(os.path.join(save_dir, ts.strftime("%Y-%m-%dT%H:%M:%S")), engine="h5netcdf")