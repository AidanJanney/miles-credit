#!/bin/bash
# Short diagnostic run of an rMOM6 _final config -- the OBC-halo path has never executed at
# scale, so this is the cheapest way to find out whether the five blocking fixes hold on real
# hardware before spending a 12-hour slot on it.
#
# Derives a throwaway config from the one you name: a handful of iterations, one epoch, a
# separate save_loc, and no checkpoint reload. The source config is not modified.
#
#   scripts/submit_rmom6_smoke.sh config/rmom6_regional_levelwise_singlestep_notendency_final.yml
#   scripts/submit_rmom6_smoke.sh <config> 200          # 200 iterations instead of 50
#
# What to check in the log, in order:
#   1. "Loss mask built from raw targets: 49.2% of 101 channels x 458 x 760 cells are valid."
#      Any other percentage, or a 457x759 grid, means the halo/mask wiring is off.
#   2. train_loss is finite and falling. A NaN on iteration 1 points at the mask or the scaler.
#   3. It reaches the end without raising. That alone clears findings 01-05.
set -euo pipefail

CONFIG="${1:?usage: $0 <config.yml> [iterations]}"
ITERS="${2:-50}"
REPO="/glade/work/ajanney/RegionalEmulation_v2/miles-credit-regional"
cd "$REPO"

SMOKE_CONF=$(mktemp /glade/derecho/scratch/ajanney/tmp/rmom6_smoke.XXXXXX.yml)

python - "$CONFIG" "$SMOKE_CONF" "$ITERS" <<'PY'
import os, sys, yaml

src, dst, iters = sys.argv[1], sys.argv[2], int(sys.argv[3])
c = yaml.safe_load(open(src))

name = os.path.basename(src).replace(".yml", "")
c["save_loc"] = f"/glade/derecho/scratch/ajanney/CREDIT_runs/_smoke_{name}"

t = c.setdefault("trainer", {})
t["epochs"] = 1
t["batches_per_epoch"] = iters
t["valid_batches_per_epoch"] = max(4, iters // 10)
t["start_epoch"] = 0
t["reload_epoch"] = False
t["load_weights"] = False
t["load_optimizer"] = False

p = c.setdefault("pbs", {})
p["job_name"] = f"smoke_{name[:28]}"
p["walltime"] = "01:00:00"

yaml.safe_dump(c, open(dst, "w"), sort_keys=False)
print(f"  save_loc: {c['save_loc']}", file=sys.stderr)
print(f"  {iters} train iters, {t['valid_batches_per_epoch']} valid, 1 epoch", file=sys.stderr)
PY

echo "  derived config: $SMOKE_CONF"
echo
# Set DRYRUN=1 to print the generated PBS script instead of submitting it.
echo "Submitting..."
"${CREDIT:-credit}" submit --cluster derecho -c "$SMOKE_CONF" \
  ${DRYRUN:+--dry-run} \
  --gpus "$(python -c "import yaml,sys; print(yaml.safe_load(open('$SMOKE_CONF'))['pbs'].get('ngpus',4))")" \
  --nodes "$(python -c "import yaml,sys; print(yaml.safe_load(open('$SMOKE_CONF'))['pbs'].get('nodes',1))")"
