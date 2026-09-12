#!/usr/bin/env bash
# Can RFLO's arm plateau be lifted, and does the feedback matrix B set where it fails?
#
#   Bonly    : one network, B swept over 5 values. Init and target stream are fixed, so
#              anything that moves in the reach profile is attributable to B alone.
#   nrec128  : wider recurrent layer, longer budget (wider nets converge slower).
#   nrec256  : same, further.
#   termloss : up-weight the last timesteps of the loss.
#
# RFLO only, behavior and hidden states only. Select cells with CELLS.
#
# Usage:  CELLS="Bonly termloss" nohup bash drivers/arm_lift.sh &>logs/arm_lift/driver.log &

source "$(dirname "$0")/_common.sh"

SEEDS=${SEEDS:-"0 1 2"}
B_SEEDS=${B_SEEDS:-"0 1 2 3 4"}
CELLS=${CELLS:-"Bonly nrec128 nrec256 termloss"}
NET=${NET:-0}

banner start "lr=$LR cells={$CELLS}"
for cell in $CELLS; do
  case "$cell" in
    # One artifact name per (fixed net) seed, so each B value needs its own directory.
    Bonly)
      for b in $B_SEEDS; do
        launch "results/arm_lift_Bonly_b${b}" "$NET" rflo --b-seed "$b" &
        throttle
      done
      ;;
    nrec128)
      for s in $SEEDS; do
        HIDDEN=128 STEPS=100000 launch results/arm_lift_nrec128 "$s" rflo &
        throttle
      done
      ;;
    nrec256)
      for s in $SEEDS; do
        HIDDEN=256 STEPS=120000 launch results/arm_lift_nrec256 "$s" rflo &
        throttle
      done
      ;;
    termloss)
      for s in $SEEDS; do
        launch results/arm_lift_termloss "$s" rflo \
          --loss terminal --terminal-weight 0.5 --terminal-k 10 &
        throttle
      done
      ;;
    *) echo "unknown cell '$cell'" >&2; exit 1 ;;
  esac
done
wait
banner done
echo "Next: python per_direction_error.py --dir results/arm_lift_<cell> --rules RFLO --per-seed"
