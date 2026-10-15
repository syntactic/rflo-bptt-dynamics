#!/usr/bin/env bash
# BPTT on the arm with its W_rec and W_in gradients projected onto span(B) before each
# step. W_out stays free, as in RFLO.
#
# RFLO's recurrent and input updates are confined near span(B). This gives BPTT the same
# confinement with exact credit, separating what the subspace costs from what RFLO's
# approximate credit costs. The default basis is B from seed 0, the matrix the fixB RFLO
# runs share, and seed s starts from the same initialization as BPTT and RFLO seed s.
#
# Usage:  WEIGHT_HISTORY=1 nohup bash drivers/arm_bptt_projected.sh &>logs/arm_bptt_projected/driver.log &

# _common.sh defaults STEPS to 40k, so the longer default has to be set first.
STEPS=${STEPS:-100000}
source "$(dirname "$0")/_common.sh"

SEEDS=${SEEDS:-"1 3 4 5 8 9 10 11"}
BASIS_SEED=${BASIS_SEED:-0}
OUT=${OUT:-"results/arm_bptt_projB${BASIS_SEED}_$((STEPS / 1000))k${WH_SUFFIX}"}

banner start "lr=$LR steps=$STEPS basis_seed=$BASIS_SEED seeds={$SEEDS} out=$OUT"
for s in $SEEDS; do
  launch "$OUT" "$s" bptt --bptt-grad-basis-seed "$BASIS_SEED" &
  throttle
done
wait
banner done
