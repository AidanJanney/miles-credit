#!/bin/bash -l
#PBS -N rmom6_levelpair_pointwise_stats
#PBS -l select=1:ncpus=36:ngpus=0:mem=300GB
#PBS -l walltime=03:00:00
#PBS -A p93300012
#PBS -q casper
#PBS -j oe
#PBS -k eod
module load conda
conda activate miles-credit-casper

# Recomputes per-(level, y, x) mean/std for preprocess_rmom6.py --level-pairs' merged 25
# levels, directly from the already-preprocessed rmom6_prognostic_<year>.zarr stores -- see
# build_rmom6_levelpair_pointwise_stats.py's docstring. Companion to
# casper_build_rmom6_levelpair_stats.sh (which produced stats_xi_levelpairs.nc, reused as-is
# here since xi has no spatial component). --scheduler threads for the same dask.distributed
# version-mismatch reason documented in that script and DASK_DISTRIBUTED_ISSUE.md.
python -u build_rmom6_levelpair_pointwise_stats.py \
  --years 2000-2019 \
  --scheduler threads \
  --n-workers 36
