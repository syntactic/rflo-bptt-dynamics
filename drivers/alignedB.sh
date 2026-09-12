#!/usr/bin/env bash
# Aligned feedback: replace RFLO's random B with the live readout weights.
#
# evaryB.sh shows the corridor needs a shared B. This asks what kind of B that has to be,
# by making the credit projection exact instead of random while leaving the eligibility
# traces untouched. The readout is ~50x smaller than a random B at init, so the default
# 'aligned-matched' rescales it to sqrt(n_rec) and changes feedback direction only;
# B_MODE=aligned is the raw variant. Aligned B is per-seed, so the one-factor contrast
# is against varyB, not fixB.
#
# PILOT=1 runs 3 seeds without weight history to check that both gain variants converge.
#
# Usage:  nohup bash drivers/alignedB.sh &>logs/alignedB/driver.log &

WEIGHT_HISTORY=${WEIGHT_HISTORY:-1}
source "$(dirname "$0")/_common.sh"

B_MODE=${B_MODE:-aligned-matched}
# Seeds that converged under both fixB and varyB, so the contrast runs on one roster.
SEEDS=${SEEDS:-"1 3 4 5 8 9 10 11 12 14 18 19 20 21 22 23 24"}
# 'aligned' is the raw readout, 'aligned-matched' the norm-matched one; keep the two
# gain variants in separate directories.
gain() { [[ "$1" == aligned ]] && echo raw || echo matched; }
OUT=${OUT:-"results/alignedB_$(gain "$B_MODE")${WH_SUFFIX}"}

if [[ "${PILOT:-0}" == "1" ]]; then
  SEEDS="1 3 4"
  banner start "pilot: raw vs matched gain, seeds={$SEEDS}"
  for mode in aligned aligned-matched; do
    for s in $SEEDS; do
      WH_FLAG="--no-weight-history" launch "results/pilot_alignedB_$(gain "$mode")" "$s" rflo --b-seed "$mode" &
      throttle
    done
  done
else
  banner start "lr=$LR steps=$STEPS b_mode=$B_MODE seeds={$SEEDS} out=$OUT"
  for s in $SEEDS; do
    launch "$OUT" "$s" rflo --b-seed "$B_MODE" &
    throttle
  done
fi
wait
banner done
echo "Next: python check_convergence.py --dir <output dir> --rule RFLO, then compare the"
echo "      surviving seeds against results/evaryB_varyB${WH_SUFFIX} (both per-seed B)."
