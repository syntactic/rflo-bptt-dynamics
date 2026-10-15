#!/usr/bin/env bash
# Shared-B RFLO on the arm, repeated over several draws of B.
#
# evaryB.sh's fixB condition shares one matrix (b_seed 0) across all nets, so its
# corridor rests on a single draw. Here each draw is shared across the same nets, giving
# a within-draw corridor per B. The default B seeds sit outside 0-24 so none of them
# coincides with a varyB (b_seed=match) draw. The b_seed 0 reference on the same nets is
# results/evaryB_fixB_wh.
#
# Usage:  nohup bash drivers/multiB.sh &>logs/multiB/driver.log &

# The corridor is measured on weight trajectories, so weight history is on by default.
WEIGHT_HISTORY=${WEIGHT_HISTORY:-1}
source "$(dirname "$0")/_common.sh"

SEEDS=${SEEDS:-"1 3 4 5 8"}
B_SEEDS=${B_SEEDS:-"101 102 103"}

banner start "lr=$LR steps=$STEPS b_seeds={$B_SEEDS} seeds={$SEEDS}"
for b in $B_SEEDS; do
  out="results/multiB_b${b}${WH_SUFFIX}"
  for s in $SEEDS; do
    launch "$out" "$s" rflo --b-seed "$b" &
    throttle
  done
done
wait
banner done
echo "Next: python check_convergence.py --dir results/multiB_b<B>${WH_SUFFIX} --rule RFLO,"
echo "      then run_subspace_geometry.py with one condition per draw plus fixB and varyB."
