#!/bin/bash -l
#PBS -N rmom6_rechunk
#PBS -l select=1:ncpus=16:ngpus=0:mem=100GB
#PBS -l walltime=01:00:00
#PBS -A p93300012
#PBS -q main
#PBS -j oe
#PBS -k eod
#PBS -J 2000-2019
module load conda
conda activate miles-credit-derecho
cd ..

# One array sub-job per year (${PBS_ARRAY_INDEX}); years are independent. Rewrites each
# year's prognostic + forcing store with a time chunk of 1 (was 30), in place and atomically
# -- see scripts/rechunk_rmom6.py for why (30x read amplification made training entirely
# dataloader-bound at ~500 s/iter). Measured ~3 min/year, so 01:00:00 is generous margin.
# Rerunning is safe: years already at time-chunk 1 are skipped unless --overwrite is set.
#
# Only needed for stores written before preprocess_rmom6.py's --time-chunk default changed
# from 30 to 1; anything preprocessed after that is already correct.
python -u scripts/rechunk_rmom6.py \
  --years "${PBS_ARRAY_INDEX}" \
  --data-dir "/glade/derecho/scratch/${USER}/rmom6_regional/preprocessed"
