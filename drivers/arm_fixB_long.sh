#!/usr/bin/env bash
# Shared-B RFLO on the arm past the 40k default budget.
#
# RFLO's arm loss sits on a shelf from about 20k to 40k steps and then falls again, so
# comparisons with BPTT and projected BPTT have to be made at equal budget. evaryB.sh
# can't be reused for this: it writes into the 40k directories, and launch() would skip
# these seeds as already done. The output directory is named after STEPS for that reason.
#
# Usage:  STEPS=300000 nohup bash drivers/arm_fixB_long.sh &>logs/arm_fixB_long/driver.log &

# _common.sh defaults STEPS to 40k, so the longer default has to be set first.
STEPS=${STEPS:-100000}
source "$(dirname "$0")/_common.sh"

SEEDS=${SEEDS:-"1 3 4 5 8 9 10 11"}
B_SEED=${B_SEED:-0}
OUT=${OUT:-"results/arm_fixB_$((STEPS / 1000))k${WH_SUFFIX}"}

banner start "lr=$LR steps=$STEPS b_seed=$B_SEED seeds={$SEEDS} out=$OUT"
for s in $SEEDS; do
  launch "$OUT" "$s" rflo --b-seed "$B_SEED" &
  throttle
done
wait
banner done
