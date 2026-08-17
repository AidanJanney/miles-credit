#!/bin/bash
# Submit a single-GPU Gen2 ocean rollout (rollout_gen2.py) to Derecho PBS.
# Usage: scripts/submit_rmom6_rollout.sh <rollout_config.yml>
# Reads pbs.{project,job_name,walltime,ncpus,ngpus,mem,queue,conda} from the config.
set -euo pipefail

CONFIG="$1"
REPO="/glade/work/ajanney/RegionalEmulation_v2/miles-credit-regional"
cd "$REPO"

# Pull pbs settings + save_forecast (for the logs dir) out of the YAML.
read -r PROJECT JOBNAME WALLTIME NCPUS NGPUS MEM QUEUE CONDA SAVEDIR < <(
python - "$CONFIG" <<'PY'
import sys, os, yaml
c = yaml.safe_load(open(sys.argv[1]))
p = c["pbs"]
save = os.path.expandvars(c["inference"]["save_forecast"])
print(p.get("project","p93300012"), p["job_name"], p["walltime"],
      p.get("ncpus",8), p.get("ngpus",1), p.get("mem","64GB"),
      p.get("queue","main"), p.get("conda","miles-credit-derecho"), save)
PY
)

LOGDIR="${SAVEDIR}/logs"
mkdir -p "$LOGDIR"

SCRIPT=$(mktemp /glade/derecho/scratch/ajanney/tmp/rollout_pbs.XXXXXX.sh)
cat > "$SCRIPT" <<EOF
#!/bin/bash
#PBS -A ${PROJECT}
#PBS -N ${JOBNAME}
#PBS -l walltime=${WALLTIME}
#PBS -l select=1:ncpus=${NCPUS}:ngpus=${NGPUS}:mem=${MEM}
#PBS -q ${QUEUE}@desched1
#PBS -j oe
#PBS -k eod
#PBS -r n
#PBS -o ${LOGDIR}
module load ncarenv/24.12 gcc/12.4.0 ncarcompilers craype cray-mpich/8.1.29 \\
            cuda/12.3.2 conda/latest cudnn/9.2.0.82-12 mkl/2025.0.1
conda activate ${CONDA}
cd ${REPO}
echo "Node: \$(hostname)  Config: ${CONFIG}"
nvidia-smi -L || true
python credit/applications/rollout_gen2.py -c ${CONFIG} -p ${NCPUS}
EOF

echo "PBS script: $SCRIPT"
echo "Logs dir  : $LOGDIR"
qsub "$SCRIPT"
