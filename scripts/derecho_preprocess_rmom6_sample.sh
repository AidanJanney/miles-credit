#!/bin/bash -l
#PBS -N rmom6_preprocess_sample
#PBS -l select=1:ncpus=16:ngpus=0:mem=64GB
#PBS -l walltime=00:30:00
#PBS -A NAML0001
#PBS -q main
#PBS -j oe
#PBS -k eod
module load conda
conda activate miles-credit-derecho
cd ..

# 15-day sample (2000-01-01..2000-01-15) of everything config/rmom6_regional.yml expects:
# prognostic + forcing state, OBC boundaries (not yet wired into the config), and the
# bridgescaler normalization stats. Safe to rerun — each step overwrites only its own output.
OUT="/glade/derecho/scratch/${USER}/rmom6_regional/preprocessed"

# preprocess_rmom6.py's Arakawa C->A regrid is a plain xarray/numpy center-average (no xgcm
# dependency) — everything below runs in the one regular credit env.
python -u scripts/preprocess_rmom6.py \
  --start-date 2000-01-01 --end-date 2000-01-16 \
  --out-dir "$OUT" --overwrite

python -u scripts/preprocess_obcs.py \
  --start-date 2000-01-01 --end-date 2000-01-16 \
  --out-dir "$OUT" --overwrite

python -u scripts/build_rmom6_scaler.py \
  --out "/glade/derecho/scratch/${USER}/rmom6_regional/scaler/ocean_bridgescaler.json"
