#!/bin/bash -l
#PBS -N rmom6_preprocess
#PBS -l select=1:ncpus=16:ngpus=0:mem=100GB
#PBS -l walltime=06:00:00
#PBS -A p93300012
#PBS -q casper
#PBS -j oe
#PBS -k eod
#PBS -m abe
#PBS -J 2000-2019
module load conda
conda activate miles-credit-casper

# One array sub-job per year (${PBS_ARRAY_INDEX}); years are independent so this
# scales to as many jobs as the queue will run concurrently (measured ~9m43s for one
# year sequentially on Casper -- 02:00:00 leaves generous margin per sub-job). Rerunning
# is safe -- process_year() skips any year whose output already exists unless --overwrite
# is set. --levels matches downsample_vgrid_carib12/vgrid_select/selected_levels_n25.json
# -- must stay identical to what build_rmom6_scaler.py / build_rmom6_pointwise_scaler.py
# were run with, see EXPERIMENT_DESIGN.md §2.
# LEVELS="3.8,5.1,7.9,9.6,92.3,130.7,155.9,186.1,222.5,266.0,318.1,380.2,453.9,541.1,643.6,763.3,1062.4,1245.3,1452.3,1684.3,1941.9,2225.1,2533.3,2865.7,3220.8"

python -u preprocess_rmom6.py \
  --years "${PBS_ARRAY_INDEX}" \
  --level-pairs \
  --out-dir "/glade/derecho/scratch/${USER}/rmom6_regional/preprocessed" \
  --overwrite
