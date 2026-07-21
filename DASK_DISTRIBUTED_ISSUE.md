# `dask.distributed` fails to import: `ImportError: cannot import name 'collections_to_dsk' from 'dask.base'`

## Environment

- Conda env: `miles-credit-casper` (`/glade/work/ajanney/conda-envs/miles-credit-casper`)
- Python: 3.11.8 (conda-forge, GCC 12.3.0)
- Platform: `Linux-6.4.0-150600.23.81-default-x86_64-with-glibc2.38` (NSF NCAR Casper)
- `dask`: **2026.3.0**
- `distributed`: **2025.3.0**
- `xarray`: 2026.2.0
- `zarr`: 3.0.6

## Reproduction

```python
import dask.distributed
```

or equivalently:

```python
from dask.distributed import Client, LocalCluster
```

## Actual result

```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/glade/work/ajanney/conda-envs/miles-credit-casper/lib/python3.11/site-packages/dask/distributed.py", line 11, in <module>
    from distributed import *  # noqa: F403
    ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/glade/work/ajanney/conda-envs/miles-credit-casper/lib/python3.11/site-packages/distributed/__init__.py", line 23, in <module>
    from distributed.actor import Actor, ActorFuture, BaseActorFuture
  File "/glade/work/ajanney/conda-envs/miles-credit-casper/lib/python3.11/site-packages/distributed/actor.py", line 15, in <module>
    from distributed.client import Future
  File "/glade/work/ajanney/conda-envs/miles-credit-casper/lib/python3.11/site-packages/distributed/client.py", line 44, in <module>
    from dask.base import collections_to_dsk
ImportError: cannot import name 'collections_to_dsk' from 'dask.base' (/glade/work/ajanney/conda-envs/miles-credit-casper/lib/python3.11/site-packages/dask/base.py)
```

## Expected result

`dask.distributed` imports successfully and `Client`/`LocalCluster` are usable, as with any
matched `dask`/`distributed` pair.

## Root cause

Version skew between `dask` (2026.3.0) and `distributed` (2025.3.0) — roughly a year apart.
`distributed.client` imports the internal helper `dask.base.collections_to_dsk`, which no longer
exists in `dask` 2026.3.0. `dask/base.py` in this env only defines `collections_to_expr` (line
413), suggesting the helper was renamed/replaced as part of dask's move to its expression-based
execution engine, and `distributed` 2025.3.0 predates that change.

## Likely fix

Upgrade `distributed` to a version released alongside `dask` 2026.3.0 (or pin both to a matched
pair) — e.g. `conda install -c conda-forge "dask=2026.3.0" "distributed=2026.3.0"` in the
`miles-credit-casper` env, or let conda/mamba re-resolve both together rather than upgrading
`dask` alone.

## Impact

Any code path in this env that needs a real dask cluster (`dask.distributed.Client`,
`LocalCluster`) is currently broken — falls back to `dask`'s built-in `threads`/`synchronous`
schedulers, which work but don't give multi-process parallelism or a dashboard. This is why
`scripts/build_rmom6_levelpair_stats.py` defaults to `--scheduler threads` instead of
`--scheduler distributed`.
