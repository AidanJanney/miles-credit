#!/bin/bash -l
#PBS -N regional_mom6_emulation
#PBS -l select=1:ncpus=8:ngpus=1:mem=128GB:gpu_type=a100_80gb
#PBS -l walltime=12:00:00
#PBS -A P93300012
#PBS -q casper
#PBS -j oe
#PBS -m bae 

module load conda
conda activate credit-casper

cd /glade/work/ajanney/miles-credit/credit/trainers

export PYTHONUNBUFFERED=1
python trainer_rMOM6.py