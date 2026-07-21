from credit.preblock.base import BasePreblock
from credit.preblock._utils import _parse_variable_selection
from credit.preblock.pointwise_stats import load_pointwise_stats, apply_pointwise_scale, get_cached_stat_entry


class PointwiseScalerTransform(BasePreblock):
    """Elementwise (per-level, per-pixel) standardization preblock.

    Like ``bridgescaler_transform``, but the mean/std vary spatially (not just per level) --
    see ``credit/preblock/pointwise_stats.py`` for why bridgescaler's scaler classes can't
    represent this. Applies ``batch[data_type][source][var_key]`` -> ``(x - mu) / sigma``
    (or the inverse), where ``mu``/``sigma`` come from a torch-saved stats file keyed by
    ``var_key`` (produced by ``scripts/build_rmom6_pointwise_scaler.py``).

    ``variables``/``data_types`` follow the same conventions as ``bridgescaler_transform``:
    an empty ``variables`` list expands to every variable present in the batch; partial paths
    (e.g. ``"rMOM6/prognostic"``) expand to everything beneath them. Expansion happens lazily
    on the first forward call.

    Config example::

        type: "pointwise_scaler"
        args:
            stats_path: "/path/to/ocean_pointwise_xi.pt"
            variables:
                - "rMOM6/prognostic/3d/thetao"
            method: "transform"
    """

    def __init__(
        self,
        stats_path: str,
        variables: list[str],
        method: str = "transform",
        data_types: list[str] = None,
    ):
        super().__init__()
        self.variables = variables
        self.variables_expanded = False
        self.method = method
        self.stats_path = stats_path
        self.data_types = data_types
        self.stats = load_pointwise_stats(stats_path)
        self._device_cache: dict = {}

    def forward(self, batch: dict) -> dict:
        if not self.variables_expanded:
            self.variables = _parse_variable_selection(self.variables, batch, self.data_types)
            self.variables_expanded = True
        batch = self._copy_batch(batch)
        data_types = self.data_types if self.data_types is not None else list(batch.keys())
        for data_type in data_types:
            if data_type not in batch:
                continue
            for source_vars in batch[data_type].values():
                for var_key in list(source_vars.keys()):
                    if var_key not in self.variables or var_key not in self.stats:
                        continue
                    tensor = source_vars[var_key]
                    cached_entry = get_cached_stat_entry(
                        self._device_cache, self.stats[var_key], var_key, tensor.dtype, tensor.device
                    )
                    source_vars[var_key] = apply_pointwise_scale(tensor, cached_entry, self.method)
        return batch
