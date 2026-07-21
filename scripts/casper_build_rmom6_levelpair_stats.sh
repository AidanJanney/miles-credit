#!/bin/bash -l
#PBS -N rmom6_levelpair_stats
#PBS -l select=1:ncpus=36:ngpus=0:mem=300GB
#PBS -l walltime=08:00:00
#PBS -A p93300012
#PBS -q casper
#PBS -j oe
#PBS -k eod
module load conda
conda activate miles-credit-casper

# Recomputes mean/std/xi for preprocess_rmom6.py --level-pairs' merged 25 levels, directly from
# the already-preprocessed rmom6_prognostic_<year>.zarr stores -- see
# build_rmom6_levelpair_stats.py's docstring for why this can't just reindex the existing
# native-level stats_{levelwise,xi}.nc. --scheduler threads (not distributed): as of writing,
# dask.distributed fails to import in miles-credit-casper (installed dask/distributed versions
# are mismatched -- `from dask.base import collections_to_dsk` ImportError). Switch to
# `--scheduler distributed` once that's fixed for better multi-process parallelism; threads
# still parallelizes zarr decompression reasonably well since that releases the GIL.
python -u build_rmom6_levelpair_stats.py \
  --years 2000-2019 \
  --scheduler threads \
  --n-workers 36
