#!/bin/bash -l
#PBS -N rmom6_io_throughput
#PBS -l select=1:ncpus=8:ngpus=0:mem=120GB
#PBS -l walltime=00:30:00
#PBS -A p93300012
#PBS -q casper
#PBS -j oe
#PBS -k eod
module load conda
conda activate miles-credit-casper

# Measures sustained dataloader throughput at thread_workers 0 vs 4, at production
# batch_size, for the singlestep _full config. This is the number that decides whether
# `thread_workers: 0` explains the ~45.8 s/iter seen overnight, or whether serialized
# I/O is only a small part of it and something else is still unaccounted for.
#
# Runs enough batches to drain the prefetch queue -- the whole point is to measure the
# rate workers can *sustain*, not the rate at which an already-full queue empties.

cd /glade/work/ajanney/RegionalEmulation_v2/miles-credit-regional
python -u scripts/io_throughput.py -c config/rmom6_regional_levelwise_singlestep_tendency_full.yml
