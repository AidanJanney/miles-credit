#!/bin/bash -l
#PBS -N rmom6_bench
#PBS -l select=1:ncpus=64:ngpus=1
#PBS -l walltime=02:00:00
#PBS -A p93300012
#PBS -q main
#PBS -j oe
# Explicit -o (and no -k) so the log lands at a known path and streams while the job runs.
# With `-k eod` PBS buffers the output and only delivers it at exit, which makes a long
# benchmark impossible to watch and is easy to lose track of.
#PBS -o /glade/derecho/scratch/ajanney/rmom6_bench.log
module load conda
conda activate miles-credit-derecho
cd /glade/work/ajanney/RegionalEmulation_v2/miles-credit-regional

# Single GPU, single rank on purpose: this measures where one iteration's wall clock goes
# (dataloader vs preblocks vs forward vs postblocks vs backward), and DDP all-reduce would
# only blur that attribution. See scripts/benchmark_rmom6.py's docstring.
#
# Runs both bench configs back to back. singlestep consumes 1 dataloader batch per
# iteration, multistep 3 -- the difference isolates per-rollout-step cost.
#
# --iters 12, not 3: with thread_workers=4 x prefetch_factor=2, eight batches are already
# queued before timing starts, so a 3-iteration singlestep run (3 batches) never blocks on
# the dataloader once and reports a prefetch-queue drain rather than sustained throughput.
# 12 iterations outruns the queue at both forecast_len values.
for cfg in singlestep multistep; do
  echo "################  ${cfg}  ################"
  python -u scripts/benchmark_rmom6.py -c "config/rmom6_bench_${cfg}.yml" --iters 12
done
