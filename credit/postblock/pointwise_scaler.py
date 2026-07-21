from credit.postblock.base import BasePostblock
from credit.preblock._utils import _parse_variable_selection  # shared utility, see credit/postblock/scaler.py
from credit.preblock.pointwise_stats import load_pointwise_stats, apply_pointwise_scale, get_cached_stat_entry


class PointwiseScalerTransform(BasePostblock):
    """Inverse of the ``pointwise_scaler`` preblock -- see that module's docstring.

    Applies elementwise (per-level, per-pixel) de-standardization to the reconstructed output
    dict at ``batch_dict[key]`` (default ``"y_processed"``), which has the form
    ``{source: {var_key: tensor}}``. Typically runs right after ``Reconstruct``, before
    physics postblocks (``ocean_clamp``, ``ocean_wet_mask``), mirroring ``bridgescaler_transform``.

    Config example::

        type: "pointwise_scaler"
        args:
            stats_path: "/path/to/ocean_pointwise_xi.pt"
            variables: []
            method: "inverse_transform"
    """

    def __init__(
        self,
        stats_path: str,
        variables: list[str],
        method: str = "inverse_transform",
        key: str = "y_processed",
    ):
        super().__init__()
        self.variables = variables
        self.variables_expanded = False
        self.method = method
        self.stats_path = stats_path
        self.key = key
        self.stats = load_pointwise_stats(stats_path)
        self._device_cache: dict = {}

    def forward(self, batch_dict: dict) -> dict:
        if not self.variables_expanded:
            wrapped = {"_": batch_dict[self.key]}
            self.variables = _parse_variable_selection(self.variables, wrapped, data_types=["_"])
            self.variables_expanded = True
        for source_vars in batch_dict[self.key].values():
            for var_key in list(source_vars.keys()):
                if var_key not in self.variables or var_key not in self.stats:
                    continue
                tensor = source_vars[var_key]
                cached_entry = get_cached_stat_entry(
                    self._device_cache, self.stats[var_key], var_key, tensor.dtype, tensor.device
                )
                source_vars[var_key] = apply_pointwise_scale(tensor, cached_entry, self.method)
        return batch_dict
