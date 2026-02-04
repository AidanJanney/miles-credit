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
                            vars_3D: ['uo', 'vo', 'thetao', 'so']
                            vars_2D: ['SSH']
                            path: <path_to_prognostic_data>
                            transform:
                                mean_path: <path_to_prognostic_mean_stats>
                                std_path: <path_to_prognostic_std_stats>
                            
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
        self.stats_dict = {}

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
                self.stats_dict[field_type] = {
                    "mean_path": d.get("transform", {}).get("mean_path", None),
                    "std_path": d.get("transform", {}).get("std_path", None),
                }
            else:
                self.file_dict[field_type] = None # default when null
                
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
        return len(self.datetimes)
    
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
        # self._open_ds_extract_fields("dynamic_forcing", t, return_data)
        # self._open_ds_extract_fields("north_boundary", t, return_data)
        # self._open_ds_extract_fields("east_boundary", t, return_data)
        # self._open_ds_extract_fields("west_boundary", t, return_data)
        # self._open_ds_extract_fields("south_boundary", t, return_data)

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
    
    def _open_ds_extract_fields(self, field_type, t, return_data, is_target=False):
        """
        opens the dataset, reshapes and concats the variables into n np array,
        packs it into the return dict if the data exists.

        Args:
             field_type (str): Field type ("prognostic", "diagnostic", etc)
             t (pd.Timestamp): Current timestamp
             return_data (dict): Dictionary of data to return
             is_target (bool): Flag for if data is x or y data
        """
        if self.file_dict[field_type]:
            with xr.open_dataset(self.file_dict[field_type][t.year]) as dataset:
                if "time" in dataset.dims:
                    print("testing _open_ds_extract_fields")
                    if isinstance(dataset.time.values[0], cftime.datetime):
                        t = self._convert_cf_time(t)
                    ds = dataset.sel(time=t)
                else:
                    ds = dataset

                ds_all_vars = ds[
                    self.var_dict[field_type]["vars_3D"]
                    + self.var_dict[field_type]["vars_2D"]
                ]
                
                # Transform data (should be flexible for pointwise or otherwise)
                if field_type != "static": # don't transorm static fields?
                    ds_all_vars = self._transform_data(ds_all_vars, field_type)

                if not is_target and field_type == "prognostic":
                    ds_all_vars = self._concat_obcs(ds_all_vars, t)
                        
                ds_3D = ds_all_vars[self.var_dict[field_type]["vars_3D"]]
                ds_2D = ds_all_vars[self.var_dict[field_type]["vars_2D"]]
                # ds_all_vars = self._mask_data(ds_3D, ds_2D)
                data_np, meta = self._reshape_and_concat(ds_3D, ds_2D)

                if is_target:
                    if field_type == "prognostic":
                        return_data["target_prognostic"] = torch.tensor(data_np).float()
                    elif field_type == "diagnostic":
                        return_data["target_diagnostic"] = torch.tensor(data_np).float()
                else:
                    return_data[field_type] = torch.tensor(data_np).float()

                return_data["metadata"][f"{field_type}_var_order"] = meta
                
    def _reshape_and_concat(self, ds_3D, ds_2D):
        """
        Stack 3D variables along level and variable, concatenate with 2D variables, and reorder dimensions. 

        Args:
            ds_3D (xr.Dataset): Xarray dataset with 3D spatial variables
            ds_2D (xr.Dataset): Xarray dataset with 2D spatial variables
        """
        data_list = []
        meta_3D, meta_2D = [], []

        if ds_3D:
            data_3D = ds_3D.to_array().stack({"level_var": ["variable", "level"]})
            meta_3D = data_3D.level_var.values.tolist()
            data_3D = np.expand_dims(data_3D.values.transpose(2, 0, 1), axis=1)
            data_list.append(data_3D)

        if ds_2D:
            data_2D = ds_2D.to_array()
            meta_2D = data_2D["variable"].values.tolist()
            data_2D = np.expand_dims(data_2D, axis=1)
            data_list.append(data_2D)

        combined_data = np.concatenate(data_list, axis=0)
        meta = meta_3D + meta_2D

        return combined_data, meta
    
    def _transform_data(self, ds, field_type):
        """Normalize data given mean/std paths from the config

        Args:
            ds (xr.Dataset): Xarray dataset to transform
            field_type (str): Field type ("prognostic", "diagnostic", etc)
        """
        if self.stats_dict[field_type]["mean_path"] and self.stats_dict[field_type]["std_path"]:
            with xr.open_dataset(self.stats_dict[field_type]["mean_path"]) as mean_ds, xr.open_dataset(self.stats_dict[field_type]["std_path"]) as std_ds:
                for var in ds.data_vars:
                    if var in mean_ds and var in std_ds:
                        da = ds[var]
                        mean_da = mean_ds[var]
                        std_da = std_ds[var]
                        ds[var] = (da - mean_da) / std_da
                    else:
                        logger.warning(f"Variable '{var}' not found in mean/std datasets for field type '{field_type}'. Skipping normalization for this variable.")
        return ds
    
    def inverse_transform_data(self, ds, field_type):
        if self.stats_dict[field_type]["mean_path"] and self.stats_dict[field_type]["std_path"]:
            with xr.open_dataset(self.stats_dict[field_type]["mean_path"]) as mean_ds, xr.open_dataset(self.stats_dict[field_type]["std_path"]) as std_ds:
                for var in ds.data_vars:
                    if var in mean_ds and var in std_ds:
                        da = ds[var]
                        mean_da = mean_ds[var]
                        std_da = std_ds[var]
                        ds[var] = da * std_da + mean_da
                    else:
                        logger.warning(f"Variable '{var}' not found in mean/std datasets for field type '{field_type}'. Skipping inverse transform for this variable.")
        else:
            logger.warning(f"Mean/std paths not provided for field type '{field_type}'. Skipping inverse transform.")
        return ds
    
    # def _mask_data(self, ds, field_type):
    #     """Apply masking to data if a mask variable is present in the dataset

    #     Args:
    #         ds (xr.Dataset): Xarray dataset to mask
    #         field_type (str): Field type ("prognostic", "diagnostic", etc)
    #     """
    #     if self.stats_dict[field_type]
    #         if mask_var_name in ds:
    #             mask = ds[mask_var_name]
    #             for var in ds.data_vars:
    #                 if var != mask_var_name:
    #                     ds[var] = ds[var].where(mask)
    #         return ds

    def _add_metadata(self, return_data, t, t_target=None):
        """
        Update metadata dictionary

        Args:
            return_data (dict): Return dictionary
            t (int): Time step
            t_target: Target time step or None
        """
        return_data["metadata"]["input_datetime"] = int(t.value)

        if self.return_target:
            return_data["metadata"]["target_datetime"] = int(t_target.value)

    def _convert_cf_time(self, ts):
        """
        Convert pandas timestamp to cftime

        Args:
            ts: pandas timestamp
        """
        cf_t = cftime.datetime(
            ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second, calendar="noleap"
        )

        return cf_t

    def _pop_and_merge_targets(self, return_data, dim=0):
        """
        Look for target diagnostic and prognostic variables. If both exist, concatenate them along specified dimension.

        Args:
            return_data: Dictionary of current data to return
            dim: Concat dimension
        """
        target_tensors = []
        for key in ("target_prognostic", "target_diagnostic"):
            if key in return_data:
                target_tensors.append(return_data.pop(key))

        if not target_tensors:
            return

        return_data["target"] = (
            target_tensors[0]
            if len(target_tensors) == 1
            else torch.cat(target_tensors, dim=dim)
        )
    
    def _concat_obcs(self, ds, t):
        """Concat all available OBCs to prognostic dataset. field_type is assumed to be "prognostic".

        Args:
            ds (_type_): _description_
        """
        for field in ['north_boundary', 'south_boundary', 'east_boundary', 'west_boundary']:
            if self.file_dict[field]:
                with xr.open_dataset(self.file_dict[field][t.year]) as obc_ds:
                    obc_ds = obc_ds.sel(time=t)
                    if 'north' in field or 'south' in field:
                        ds = xr.concat([ds, obc_ds], dim="latitude")
                    elif 'east' in field or 'west' in field:
                        ds = xr.concat([ds, obc_ds], dim="longitude")
        return ds