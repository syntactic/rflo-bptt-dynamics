# Shared setup for the training drivers. Source it, don't run it.
#
# Every driver launches one seed per process and throttles to MAX at a time, so an
# interrupted campaign can simply be re-launched: launch() skips any artifact already
# on disk.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  for base in "$HOME/miniconda3" "$HOME/anaconda3"; do
    if [[ -f "$base/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1091
      source "$base/etc/profile.d/conda.sh"
      conda activate "${CONDA_ENV:-neuromatch}"
      break
    fi
  done
fi

# One thread per process, so MAX concurrent runs map 1:1 onto cores instead of
# oversubscribing them.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

EFFECTOR=${EFFECTOR:-RigidTendonArm26}
LR=${LR:-0.025}
STEPS=${STEPS:-40000}
HIDDEN=${HIDDEN:-64}
MAX=${MAX:-4}

# Weight history costs ~0.66 GB per arm seed at 40k steps and is only needed for the
# weight-geometry channel. Behavior-only and weight-history artifacts share a filename,
# so the "_wh" suffix keeps them in separate directories and the skip logic honest.
WH_FLAG="--no-weight-history"
WH_SUFFIX=""
if [[ "${WEIGHT_HISTORY:-0}" == "1" ]]; then
  WH_FLAG=""
  WH_SUFFIX="_wh"
fi

DRIVER="$(basename "${BASH_SOURCE[1]:-driver}" .sh)"
LOGDIR="logs/$DRIVER"
mkdir -p "$LOGDIR"

# Keep the machine awake for long overnight campaigns (macOS only).
CAFF=()
[[ "${CAFFEINATE:-1}" == "1" ]] && command -v caffeinate >/dev/null && CAFF=(caffeinate -i)

# launch <outdir> <seed> <rule> [extra run_experiment.py flags...]
launch() {
  local out=$1 seed=$2 rule=$3
  shift 3
  local tag="$(basename "$out")_seed${seed}_${rule}"
  local suffixes=("$(tr '[:lower:]' '[:upper:]' <<<"$rule")")
  [[ "$rule" == "both" ]] && suffixes=(BPTT RFLO)
  local sfx have=1
  for sfx in "${suffixes[@]}"; do
    [[ -f "$out/${EFFECTOR}_seed${seed}_${sfx}.pt" ]] || have=0
  done
  if ((have)); then
    echo "[skip] $tag (already on disk)"
    return 0
  fi
  mkdir -p "$out"
  echo "[start $(date +%H:%M:%S)] $tag"
  # shellcheck disable=SC2086
  ${CAFF[@]+"${CAFF[@]}"} python -u run_experiment.py "$EFFECTOR" --seeds "$seed" --lr "$LR" \
    --num-steps "$STEPS" --hidden-units "$HIDDEN" --rule "$rule" $WH_FLAG \
    --output-dir "$out" "$@" >"$LOGDIR/${tag}.log" 2>&1
  echo "[done  $(date +%H:%M:%S)] $tag (exit $?)"
}

# Block until a slot frees up. Call after backgrounding a launch.
throttle() {
  while (($(jobs -rp | wc -l) >= MAX)); do sleep 10; done
}

banner() {
  echo "=== $DRIVER $1 $(date) : effector=$EFFECTOR max=$MAX ${2:-} ==="
}
