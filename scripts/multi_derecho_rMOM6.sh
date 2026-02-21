#!/bin/bash -l
#PBS -N derecho_regional_mom6_emulation
#PBS -l select=1:ncpus=64:ngpus=4
#PBS -l walltime=2:00:00
#PBS -A P93300012
#PBS -q main
#PBS -j oe
#PBS -m bae 

module purge
module load ncarenv/24.12
module reset
module load gcc craype cray-mpich cuda cudnn conda
module load mkl # necessary for pytorch

conda activate credit-derecho

# cd /glade/work/ajanney/miles-credit/credit/trainers

# export PYTHONUNBUFFERED=1
# python trainer_rMOM6.py

cd /glade/work/ajanney/miles-credit/
python save_metadata.py