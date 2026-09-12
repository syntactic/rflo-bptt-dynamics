#!/usr/bin/env bash
# Hold the network initialization fixed and vary only B.
#
# The mirror of evaryB.sh, both anchored at (net 0, B 0): evaryB varies the net at fixed
# B, this varies B at a fixed net. Comparing how much the reach profile moves under each
# attributes the direction-specific deficit to B against initialization. A second anchor
# net guards against the first one being atypical.
#
# Artifact names key on the (fixed) net seed, so every B value needs its own directory.
# Behavior only: the question is which reach directions fail, which needs FT and goal.
#
# Usage:  nohup bash drivers/fixnet.sh &>logs/fixnet/driver.log &

source "$(dirname "$0")/_common.sh"

NETS=${NETS:-"0 1"}
B_SEEDS=${B_SEEDS:-"0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"}

banner start "lr=$LR steps=$STEPS nets={$NETS} b_seeds={$B_SEEDS}"
for net in $NETS; do
  echo "--- anchor net=$net ---"
  for b in $B_SEEDS; do
    launch "results/fixNet_net${net}_b${b}${WH_SUFFIX}" "$net" rflo --b-seed "$b" &
    throttle
  done
  wait  # finish one anchor before the next, so an interrupted sweep leaves net 0 complete
done
banner done
echo "Next: python per_direction_error.py --dir results/fixNet_net0_b<b> --rules RFLO"
