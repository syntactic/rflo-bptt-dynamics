#!/usr/bin/env bash
# Sensory-noise condition: proprioceptive and visual feedback noise during training.
#
# Vision noise is in the effector's own position units, so a shared absolute value would
# be a different relative perturbation per effector. It is scaled to ~1% of each reach
# distance (arm 0.1 m, point mass 0.5). Proprioception is normalized muscle length and
# velocity, so the same std applies to both. Evaluation rollouts stay clean.
#
# Usage:  nohup bash drivers/noisy.sh &>logs/noisy/driver.log &

LR=${LR:-0.05}
STEPS=${STEPS:-8000}
source "$(dirname "$0")/_common.sh"

SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9 10 11 12 13 14"}
PROP=${PROP:-1e-2}
OUT=${OUT:-results/noisy}

banner start "lr=$LR steps=$STEPS seeds={$SEEDS} out=$OUT"
for eff in ${EFFECTORS:-"ReluPointMass24 RigidTendonArm26"}; do
  EFFECTOR="$eff"
  [[ "$eff" == ReluPointMass24 ]] && VIS=5e-3 || VIS=1e-3
  for s in $SEEDS; do
    launch "$OUT" "$s" both --proprioception-noise "$PROP" --vision-noise "$VIS" &
    throttle
  done
  wait
done
banner done
echo "Next: python summarize_behavior.py --dir $OUT --n-seeds 15"
