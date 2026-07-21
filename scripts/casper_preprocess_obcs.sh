#!/bin/bash -l
#PBS -N rmom6_preprocess_obcs
#PBS -l select=1:ncpus=8:ngpus=0:mem=128GB
#PBS -l walltime=06:00:00
#PBS -A p93300012
#PBS -q main
#PBS -j oe
#PBS -k eod
#PBS -m bae
module load conda
conda activate miles-credit-casper

# Full available OBC record (all 4 boundaries), companion to derecho_preprocess_rmom6.sh's
# array job. Not split per-year (unlike the prognostic/forcing job) -- preprocess_obcs.py
# processes all 4 boundary segments in one fast pass over the whole date range, no --levels
# needed (OBC level selection happens at read-time in credit.postblock.ocean_obc_nudge, not
# preprocessing time -- see that module's docstring).
python -u preprocess_obcs.py \
  --out-dir "/glade/derecho/scratch/${USER}/rmom6_regional/preprocessed" \
  --level-pairs \
  --overwrite
