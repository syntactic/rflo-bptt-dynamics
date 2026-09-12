#!/usr/bin/env bash
# BPTT on the arm, matched to the RFLO baseline (lr=0.025, 40k steps, 64 units).
#
# This is the shared-init partner for the rule contrast: seed s here starts from the
# same initialization as seed s under RFLO, which is what makes the paired permutation
# over seeds valid. Run it with WEIGHT_HISTORY=1 to feed the weight channel.
#
# Usage:  WEIGHT_HISTORY=1 nohup bash drivers/arm_bptt.sh &>logs/arm_bptt/driver.log &

source "$(dirname "$0")/_common.sh"

SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18"}
OUT=${OUT:-"results/arm_bptt_40k${WH_SUFFIX}"}

banner start "lr=$LR steps=$STEPS seeds={$SEEDS} out=$OUT"
for s in $SEEDS; do
  launch "$OUT" "$s" bptt &
  throttle
done
wait
banner done
echo "Next: python check_convergence.py --dir $OUT --rule BPTT"
