#!/bin/bash -l
# Submit the levelwise normalization x multistep x tendency-normalization-ON experiment
# (config/rmom6_regional_levelwise_multistep_tendency.yml) as a Casper PBS batch job via `credit submit`.
# See EXPERIMENT_DESIGN.md for the full 8-config grid design and the "Multi-GPU" note.
#
# Currently points at the 15-day SAMPLE smoke-test config -- swap in a full-run config
# (see EXPERIMENT_DESIGN.md's "Graduating to the full run") once the 20-year preprocessing
# is done; resources/walltime below come from that config's own `pbs:` block.
#
# Usage:
#   ./submit_rmom6_levelwise_multistep_tendency.sh                       # single GPU, resources from pbs: block
#   ./submit_rmom6_levelwise_multistep_tendency.sh --gpus 4               # 4-GPU DDP data-parallel run
#   ./submit_rmom6_levelwise_multistep_tendency.sh --dry-run              # print the PBS script, don't submit
set -euo pipefail
cd "$(dirname "$0")/.."
credit submit --cluster casper -c config/rmom6_regional_levelwise_multistep_tendency.yml "$@"
