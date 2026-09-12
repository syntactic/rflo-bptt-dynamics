#!/usr/bin/env bash
# Learning-rate sweep for one effector, both rules.
#
# RFLO's online update needs a smaller step than BPTT, so a shared learning rate can
# look like a capability gap when it is only a step-size mismatch. LRS pairs each rate
# with its own step budget, since lower rates need longer to flatten.
#
# Point mass (the rules meet at 0.01) and arm (RFLO plateaus at every rate):
#   EFFECTOR=ReluPointMass24 LRS="0.05:20000 0.025:40000 0.01:70000" bash drivers/lr_sweep.sh
#   EFFECTOR=RigidTendonArm26 LRS="0.025:40000 0.01:70000" SEEDS="0 1 2" bash drivers/lr_sweep.sh

source "$(dirname "$0")/_common.sh"

SEEDS=${SEEDS:-"0 1 2"}
LRS=${LRS:-"0.025:40000 0.01:70000"}
TAG=${TAG:-$([[ "$EFFECTOR" == ReluPointMass24 ]] && echo pm || echo arm)}

banner start "seeds={$SEEDS} lrs={$LRS}"
for cell in $LRS; do
  LR="${cell%%:*}"
  STEPS="${cell##*:}"
  out="results/${TAG}_lr${LR}${WH_SUFFIX}"
  for s in $SEEDS; do
    launch "$out" "$s" both &
    throttle
  done
done
wait
banner done
echo "Next: python summarize_behavior.py --dir results/${TAG}_lr<rate>${WH_SUFFIX} --seeds $SEEDS"
