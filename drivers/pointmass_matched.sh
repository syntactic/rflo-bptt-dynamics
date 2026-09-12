#!/usr/bin/env bash
# Point mass at lr=0.01, N=15, with weight history: the negative control.
#
# At this shared rate BPTT and RFLO reach the same accuracy, so any weight-geometry
# difference cannot be blamed on one rule simply reaching further. Artifacts are
# ~1.1 GB per rule per seed (~34 GB for the set).
#
# Usage:  nohup bash drivers/pointmass_matched.sh &>logs/pointmass_matched/driver.log &

EFFECTOR=${EFFECTOR:-ReluPointMass24}
LR=${LR:-0.01}
STEPS=${STEPS:-70000}
WEIGHT_HISTORY=${WEIGHT_HISTORY:-1}
source "$(dirname "$0")/_common.sh"

SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9 10 11 12 13 14"}
OUT=${OUT:-results/pm_lr0.01_N15}

banner start "lr=$LR steps=$STEPS seeds={$SEEDS} out=$OUT"
for s in $SEEDS; do
  launch "$OUT" "$s" both &
  throttle
done
wait
banner done
echo "Next: python summarize_behavior.py --dir $OUT --n-seeds 15 --per-seed"
