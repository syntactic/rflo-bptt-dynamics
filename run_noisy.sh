#!/usr/bin/env bash
# Phase 2: N=15 noisy condition, seeds matched to the clean set (paired design).
# Noise scaled per-effector:
#   - proprioception is normalized muscle length/velocity (~O(1)) -> same std both effectors
#   - vision is in position units -> scales with reach (arm 0.1 m -> 1e-3; point mass 0.5-box -> 5e-3)
# The deterministic-eval fix keeps saved H a clean readout even though training is noisy.
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate neuromatch

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p results/noisy logs

run_wave() {
  local effector="$1" vis="$2" prop="$3" tag="$4"
  shift 4
  local seeds=("$@")
  echo "--- ${effector} noisy seeds ${seeds[*]} (prop=${prop}, vision=${vis}) ---"
  python run_experiment.py "${effector}" --seeds "${seeds[@]}" \
    --lr 0.05 --proprioception-noise "${prop}" --vision-noise "${vis}" \
    --output-dir results/noisy >"logs/${tag}.log" 2>&1
}

caffeinate -i bash -c '
'"$(declare -f run_wave)"'
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Point mass: vision 5e-3 (~1% of 0.5 reach), prop 1e-2
run_wave ReluPointMass24 5e-3 1e-2 pm_noisy_0-4   0 1 2 3 4 &
run_wave ReluPointMass24 5e-3 1e-2 pm_noisy_5-9   5 6 7 8 9 &
run_wave ReluPointMass24 5e-3 1e-2 pm_noisy_10-14 10 11 12 13 14 &
wait
echo "=== point mass noisy done ==="

# Arm: vision 1e-3 (~1% of 0.1 m reach), prop 1e-2
run_wave RigidTendonArm26 1e-3 1e-2 arm_noisy_0-4   0 1 2 3 4 &
run_wave RigidTendonArm26 1e-3 1e-2 arm_noisy_5-9   5 6 7 8 9 &
run_wave RigidTendonArm26 1e-3 1e-2 arm_noisy_10-14 10 11 12 13 14 &
wait
echo "=== arm noisy done ==="
'
