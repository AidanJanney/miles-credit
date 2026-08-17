import torch
import numpy as np
from datetime import datetime
from credit.losses.weighted_loss import latitude_weights
from credit.parallel.domain import shard_lat_weights


def _w_lat_for_target(metrics_obj, target):
    """Sharded, device-resident latitude weights, cached per (H, dtype, device).

    The shard and the host-to-device copy are invariant for a run, so caching
    avoids re-doing both on every metrics call.
    """
    if metrics_obj.w_lat is None:
        return 1.0
    key = (target.shape[-2], target.dtype, target.device)
    if getattr(metrics_obj, "_w_lat_key", None) != key:
        w = shard_lat_weights(metrics_obj.w_lat, target.shape[-2])
        metrics_obj._w_lat_cached = w.to(dtype=target.dtype, device=target.device)
        metrics_obj._w_lat_key = key
    return metrics_obj._w_lat_cached


class LatWeightedMetrics:
    def __init__(self, conf, training_mode=True):
        self.conf = conf
        atmos_vars = conf["data"]["variables"]
        surface_vars = conf["data"]["surface_variables"]
        diag_vars = conf["data"]["diagnostic_variables"]

        levels = conf["model"]["levels"] if "levels" in conf["model"] else conf["model"]["frames"]

        self.vars = [f"{v}_{k}" for v in atmos_vars for k in range(levels)]
        self.vars += surface_vars
        self.vars += diag_vars

        self.w_lat = None
        if conf["loss"]["use_latitude_weights"]:
            self.w_lat = latitude_weights(conf)[:, 10].unsqueeze(0).unsqueeze(-1)

        # DO NOT apply these weights during metrics computations, only on the loss during
        self.w_var = None
        if training_mode:
            self.ensemble_size = conf["trainer"].get("ensemble_size", 1)  # default value of 1 if not set
        else:
            self.ensemble_size = conf["predict"].get("ensemble_size", 1)

    def _get_w_lat(self, target):
        return _w_lat_for_target(self, target)

    def __call__(self, pred, y, clim=None, transform=None, forecast_datetime=0, mask=None):
        # forecast_datetime is passed for interface consistency but not used here

        # ``mask``: optional (1, C, H, W) float tensor, 1 where the dataset defines a target.
        # When given, every metric is computed over those cells only. Without it the reductions
        # run over the full grid, which for a regional ocean domain means roughly half the cells
        # are land that ``fill_values`` set to a constant 0 in both pred and y -- predicted
        # perfectly by construction, and therefore free skill in the average. Measured on the
        # rMOM6 carib12 grid (50.75% land): mae reads ~2.03x better than the ocean-only value
        # and rmse ~1.44x, while acc is barely affected (land sits at the field mean, so it adds
        # little to a centred correlation). Off unless the caller passes a mask, so existing
        # runs and other CREDIT configs keep their historical numbers.
        if transform is not None:
            pred = transform(pred)
            y = transform(y)

        # Get latitude and variable weights
        w_lat = self._get_w_lat(pred)
        w_var = self.w_var.to(dtype=pred.dtype, device=pred.device) if self.w_var is not None else 1.0
        w = w_var * w_lat

        if clim is not None:
            clim = clim.to(device=y.device).unsqueeze(0)
            pred = pred - clim
            y = y - clim

        loss_dict = {}
        with torch.no_grad():
            # calculate ensemble mean, if ensemble_size=1, does nothing
            if self.ensemble_size > 1:
                pred = pred.view(y.shape[0], self.ensemble_size, *y.shape[1:])  # b, ensemble, c, t, lat, lon
                std_dev = (
                    torch.std(pred, dim=1) * (self.ensemble_size + 1) / (self.ensemble_size - 1)
                )  # std dev of ensemble
                pred = pred.mean(dim=1)

            error = pred - y

            # Add epsilon to avoid division by zero
            epsilon = 1e-7

            # Reduce over every dim except the channel dim (dim=1), batched across all
            # variables at once instead of a Python loop with one GPU reduction (and,
            # via the old .cpu().item() aggregation, one forced GPU sync) per variable.
            # Same math as the loop version, just computed for all channels together.
            dims_no_c = tuple(d for d in range(pred.dim()) if d != 1)

            if mask is None:
                pred_prime = pred - pred.mean(dim=dims_no_c, keepdim=True)
                y_prime = y - y.mean(dim=dims_no_c, keepdim=True)

                denom = (
                    torch.sqrt(torch.sum(w * pred_prime**2, dim=dims_no_c) * torch.sum(w * y_prime**2, dim=dims_no_c))
                    + epsilon
                )
                acc = torch.sum(w * pred_prime * y_prime, dim=dims_no_c) / denom

                # rmse: mean over (H, W) first (matching the loop version's dim=(-2, -1)
                # inner mean), then mean over whatever dims remain except channel.
                rmse_inner = torch.sqrt(torch.mean(error**2 * w, dim=(-2, -1)))
                dims_no_c_reduced = tuple(d for d in range(rmse_inner.dim()) if d != 1)
                rmse = rmse_inner.mean(dim=dims_no_c_reduced)

                mse = torch.mean(error**2 * w, dim=dims_no_c)
                mae = torch.mean(torch.abs(error) * w, dim=dims_no_c)
            else:
                # Same statistics, restricted to valid cells. Means become sums over the mask
                # divided by the valid-cell count, so an invalid cell contributes nothing
                # rather than contributing a zero -- the distinction that makes mae/rmse read
                # ~2x better than they are on this domain.
                m = mask.to(dtype=pred.dtype, device=pred.device)
                wm = w * m
                # Valid cells per channel, and the same count shaped for keepdim division.
                n = m.expand_as(pred).sum(dim=dims_no_c).clamp(min=1.0)
                n_keep = n.view([1, -1] + [1] * (pred.dim() - 2))

                pred_prime = (pred - (pred * m).sum(dim=dims_no_c, keepdim=True) / n_keep) * m
                y_prime = (y - (y * m).sum(dim=dims_no_c, keepdim=True) / n_keep) * m
                denom = (
                    torch.sqrt(torch.sum(wm * pred_prime**2, dim=dims_no_c) * torch.sum(wm * y_prime**2, dim=dims_no_c))
                    + epsilon
                )
                acc = torch.sum(wm * pred_prime * y_prime, dim=dims_no_c) / denom

                # rmse keeps the unmasked version's shape of reduction: a per-(sample, channel)
                # spatial mean, square-rooted, then averaged over the remaining dims.
                n_hw = m.expand_as(pred).sum(dim=(-2, -1)).clamp(min=1.0)
                rmse_inner = torch.sqrt((error**2 * wm).sum(dim=(-2, -1)) / n_hw)
                dims_no_c_reduced = tuple(d for d in range(rmse_inner.dim()) if d != 1)
                rmse = rmse_inner.mean(dim=dims_no_c_reduced)

                mse = (error**2 * wm).sum(dim=dims_no_c) / n
                mae = (torch.abs(error) * wm).sum(dim=dims_no_c) / n

            stacked = [acc, rmse, mse, mae]
            if self.ensemble_size > 1:
                std_inner = torch.sqrt(torch.mean(std_dev**2 * w, dim=(-2, -1)))
                dims_no_c_std = tuple(d for d in range(std_inner.dim()) if d != 1)
                stacked.append(std_inner.mean(dim=dims_no_c_std))

            # Single GPU->CPU sync for everything, instead of one per variable per metric.
            stacked_np = torch.stack(stacked).detach().cpu().numpy()
            acc_vals, rmse_vals, mse_vals, mae_vals = stacked_np[0], stacked_np[1], stacked_np[2], stacked_np[3]
            if self.ensemble_size > 1:
                std_vals = stacked_np[4]

            for i, var in enumerate(self.vars):
                loss_dict[f"acc_{var}"] = float(acc_vals[i])
                loss_dict[f"rmse_{var}"] = float(rmse_vals[i])
                loss_dict[f"mse_{var}"] = float(mse_vals[i])
                loss_dict[f"mae_{var}"] = float(mae_vals[i])
                if self.ensemble_size > 1:
                    loss_dict[f"std_{var}"] = float(std_vals[i])

        # Calculate metrics averages
        loss_dict["acc"] = float(np.mean(acc_vals))
        loss_dict["rmse"] = float(np.mean(rmse_vals))
        loss_dict["mse"] = float(np.mean(mse_vals))
        loss_dict["mae"] = float(np.mean(mae_vals))
        if self.ensemble_size > 1:
            loss_dict["std"] = float(np.mean(std_vals))

        return loss_dict


class LatWeightedMetricsClimatology:
    def __init__(self, conf, climatology=None):
        self.conf = conf
        self.climatology = climatology  # xarray Dataset with climatology data

        atmos_vars = conf["data"]["variables"]
        surface_vars = conf["data"]["surface_variables"]
        diag_vars = conf["data"]["diagnostic_variables"]

        levels = conf["model"]["levels"] if "levels" in conf["model"] else conf["model"]["frames"]

        self.vars = [f"{v}_{k}" for v in atmos_vars for k in range(levels)]

        self.vars += surface_vars
        self.vars += diag_vars
        self.acc_vars = surface_vars + diag_vars

        self.w_lat = None
        if conf["loss"]["use_latitude_weights"]:
            self.w_lat = latitude_weights(conf)[:, 10].unsqueeze(0).unsqueeze(-1)

        # DO NOT apply these weights during metrics computations, only on the loss during
        self.w_var = None

    def _get_w_lat(self, target):
        return _w_lat_for_target(self, target)

    def get_climatology(self, forecast_datetime, variable):
        """Extract the climatology for the given forecast datetime and variable."""
        if isinstance(forecast_datetime, datetime):
            pass
        elif isinstance(forecast_datetime, int):
            forecast_datetime = datetime.utcfromtimestamp(forecast_datetime)  # Assumes integer datetime
        dayofyear = forecast_datetime.timetuple().tm_yday
        hour = forecast_datetime.hour

        # Extract climatology slice from xarray dataset
        climatology_slice = self.climatology[variable].sel(dayofyear=dayofyear, hour=hour, method="nearest")
        # Convert to PyTorch tensor
        return torch.tensor(climatology_slice.values, dtype=torch.float32)

    def __call__(self, pred, y, extras=None, transform=None, forecast_datetime=None):
        if transform is not None:
            pred = transform(pred)
            y = transform(y)

        # Get latitude and variable weights to device
        w_lat = self._get_w_lat(pred)
        w_var = self.w_var.to(dtype=pred.dtype, device=pred.device) if self.w_var is not None else 1.0

        loss_dict = {}
        with torch.no_grad():
            anomaly_scores = False
            if self.climatology and forecast_datetime:
                loss_dict = self.acc(
                    loss_dict,
                    pred,
                    y,
                    extras,
                    transform,
                    forecast_datetime,
                    w_var,
                    w_lat,
                )
                anomaly_scores = True

            # Compute RMSE, MSE, MAE for all vars
            error = pred - y
            for i, var in enumerate(self.vars):
                loss_dict[f"rmse_{var}"] = self.rmse(error[:, i], w_lat, w_var)
                loss_dict[f"mse_{var}"] = self.mse(error[:, i], w_lat, w_var)
                loss_dict[f"mae_{var}"] = self.mae(error[:, i], w_lat, w_var)
                if extras is not None:
                    for k, v in extras.items():
                        loss_dict[f"{k}_{var}"] = (v[:, i] * w_lat * w_var).mean()

            # Compute average metrics
            if anomaly_scores:
                loss_dict["acc"] = np.mean([loss_dict[k].cpu().item() for k in loss_dict.keys() if "acc_" in k])
            loss_dict["rmse"] = np.mean([loss_dict[k].cpu().item() for k in loss_dict.keys() if "rmse_" in k])
            loss_dict["mse"] = np.mean([loss_dict[k].cpu().item() for k in loss_dict.keys() if "mse_" in k])
            loss_dict["mae"] = np.mean([loss_dict[k].cpu().item() for k in loss_dict.keys() if "mae_" in k])

        return loss_dict

    def acc(self, loss_dict, pred, y, extras, transform, forecast_datetime, w_var, w_lat):
        # Compute ACC for acc_vars using anomalies
        anomalies_pred = []
        anomalies_y = []
        acc_pred = pred
        acc_y = y

        # Get the list of variables from the climatology file
        clim_vars = list(self.climatology.data_vars)

        # Ensure self.acc_vars is in the same order as clim_vars
        ordered_acc_vars = [var for var in clim_vars if var in self.vars]

        # Reorder acc_pred and acc_y to match ordered_acc_vars
        indices = [self.acc_vars.index(var) for var in ordered_acc_vars]
        acc_pred = acc_pred[:, indices]
        acc_y = acc_y[:, indices]

        # Compute anomalies
        for i, var in enumerate(ordered_acc_vars):
            clim = self.get_climatology(forecast_datetime, var).to(dtype=pred.dtype, device=pred.device).unsqueeze(0)
            anomalies_pred.append(acc_pred[:, i] - clim)
            anomalies_y.append(acc_y[:, i] - clim)

        anomalies_pred = torch.stack(anomalies_pred, dim=1)
        anomalies_y = torch.stack(anomalies_y, dim=1)

        for i, var in enumerate(self.acc_vars):
            pred_prime = anomalies_pred[:, i] - torch.mean(anomalies_pred[:, i])
            y_prime = anomalies_y[:, i] - torch.mean(anomalies_y[:, i])

            # Offset the denominator incase its zero.
            denominator = torch.sqrt(torch.sum(w_var * w_lat * pred_prime**2) * torch.sum(w_var * w_lat * y_prime**2))
            denominator = torch.maximum(denominator, torch.tensor(1e-8, device=denominator.device))
            loss_dict[f"acc_{var}"] = torch.sum(w_var * w_lat * pred_prime * y_prime) / denominator
        return loss_dict

    def rmse(self, error, w_lat, w_var):
        return torch.mean(torch.sqrt(torch.mean(error**2 * w_lat * w_var, dim=(-2, -1))))

    def mse(self, error, w_lat, w_var):
        return (error**2 * w_lat * w_var).mean()

    def mae(self, error, w_lat, w_var):
        return (torch.abs(error) * w_lat * w_var).mean()


class LatWeightedMetricsEnsemble:
    """
    metrics for rollout_ens_batcher. will output full xarrays of rmse, std etc
    """

    def __init__(self, conf, training_mode=True):
        self.conf = conf
        atmos_vars = conf["data"]["variables"]
        surface_vars = conf["data"]["surface_variables"]
        diag_vars = conf["data"]["diagnostic_variables"]

        levels = conf["model"]["levels"] if "levels" in conf["model"] else conf["model"]["frames"]

        self.vars = [f"{v}_{k}" for v in atmos_vars for k in range(levels)]
        self.vars += surface_vars
        self.vars += diag_vars

        self.w_lat = None
        if conf["loss"]["use_latitude_weights"]:
            self.w_lat = latitude_weights(conf)[:, 10].unsqueeze(0).unsqueeze(-1)

        # DO NOT apply these weights during metrics computations, only on the loss during
        self.w_var = None
        if training_mode:
            self.ensemble_size = conf["trainer"].get("ensemble_size", 1)  # default value of 1 if not set
        else:
            self.ensemble_size = conf["predict"].get("ensemble_size", 1)

    def __call__(self, pred, y, clim=None, transform=None, forecast_datetime=0):
        # pred is of shape (1, ensemble_size, c, t, lat, lon)
        # we are interested in gridcell-wise: ens mean, rmse, spread
        # TODO: spectrum
        # forecast_datetime is passed for interface consistency but not used here

        if transform is not None:
            pred = transform(pred)
            y = transform(y)

        # Get latitude and variable weights
        # w_lat = (
        #     self.w_lat.to(dtype=pred.dtype, device=pred.device)
        #     if self.w_lat is not None
        #     else 1.0
        # )
        # w_var = (
        #    self.w_var.to(dtype=pred.dtype, device=pred.device)
        #    if self.w_var is not None
        #    else 1.0
        # )

        if clim is not None:
            clim = clim.to(device=y.device).unsqueeze(0)
            pred = pred - clim
            y = y - clim

        loss_dict = {}
        with torch.no_grad():
            pred = pred.view(y.shape[0], self.ensemble_size, *y.shape[1:])  # b, ensemble, c, t, lat, lon

            loss_dict["ens_std"] = (
                torch.std(pred, dim=1) * (self.ensemble_size + 1) / (self.ensemble_size - 1)
            )  # std dev of ensemble for each gridcell/variable

            # compute ensemble mean
            pred = pred.mean(dim=1)  # b, c, t, lat, lon
            loss_dict["ens_mean"] = pred
            loss_dict["ens_rmse"] = torch.sqrt((pred - y) ** 2)

        return loss_dict


if __name__ == "__main__":
    import yaml
    import logging
    import xarray as xr
    from credit.parser import credit_main_parser

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Open an example config
    with open("../config/example-v2026.1.0.yml") as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    conf = credit_main_parser(conf, parse_training=True, parse_predict=False, print_summary=False)

    # Climatology file
    climatology_data = xr.open_dataset(conf["predict"]["climatology"])

    # Make some fake data

    true = torch.tensor(np.random.rand(1, 71, 640, 1280), dtype=torch.float32)
    pred = torch.tensor(np.random.rand(1, 71, 640, 1280), dtype=torch.float32)

    logger.info("Computing metrics. ACC without a climatology")

    # Initialize the metrics class with the climatology data
    metrics = LatWeightedMetrics(conf=conf)

    # Compute metrics
    results = metrics(pred, true)

    # Display results
    for key, value in results.items():
        print(f"{key}: {value}")

    # Comptue metrics, and ACC correctly.

    logger.info("Computing metrics. ACC with a climatology")

    # Initialize the metrics class with the climatology data
    metrics = LatWeightedMetricsClimatology(conf=conf, climatology=climatology_data)

    # Define a forecast datetime (should align with the climatology dataset)
    forecast_datetime = datetime(2024, 6, 15, 12)  # Example forecast datetime

    # Compute metrics
    results = metrics(pred, true, forecast_datetime=forecast_datetime)

    # Display results
    for key, value in results.items():
        print(f"{key}: {value}")
