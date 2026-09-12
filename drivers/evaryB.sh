#!/usr/bin/env bash
# Does sharing one fixed feedback matrix B pull independently initialized RFLO
# networks into the same region of solution space?
#
#   fixB  : every seed uses B drawn from seed 0   (--b-seed 0)
#   varyB : each seed draws its own B             (--b-seed match)
#
# Both arms run over the same network seeds, so B-sharing is the only factor that
# moves. RFLO only; BPTT has no feedback matrix. WEIGHT_HISTORY=1 for the weight
# channel (~0.66 GB/seed), off for behavior and hidden states alone (~1 MB/seed).
#
# Usage:  nohup bash drivers/evaryB.sh &>logs/evaryB/driver.log &

source "$(dirname "$0")/_common.sh"

SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9 10 11 12 13 14"}

banner start "lr=$LR steps=$STEPS seeds={$SEEDS}"
for cond in "0:results/evaryB_fixB" "match:results/evaryB_varyB"; do
  bseed="${cond%%:*}"
  out="${cond##*:}${WH_SUFFIX}"
  for s in $SEEDS; do
    launch "$out" "$s" rflo --b-seed "$bseed" &
    throttle
  done
done
wait
banner done
echo "Next: python check_convergence.py --dir results/evaryB_fixB${WH_SUFFIX} --rule RFLO (and _varyB),"
echo "      then run_condition_dsa.py (activation) and run_subspace_geometry.py (weights)."
