#!/usr/bin/env bash
set -euo pipefail

# Transfer-learning A/B: scratch vs init-from-3x3 for larger boards.
#
# Artifacts are written under research_runs/ (gitignored).
#
# Example:
#   SEED=20260209 BOARD=5 K=4 ./scripts/transfer_ab.sh

SEED="${SEED:-20260209}"
BOARD="${BOARD:-5}"
K="${K:-4}"
ITERS_3X3="${ITERS_3X3:-100}"
ITERS_BIG="${ITERS_BIG:-100}"
SKIP_3X3="${SKIP_3X3:-0}"

OUTDIR="research_runs/transfer_ab/seed_${SEED}/b${BOARD}_k${K}"
mkdir -p "${OUTDIR}"

CNN_3X3="${OUTDIR}/cnn_3x3k3_seed${SEED}.bin"
SCRATCH="${OUTDIR}/cnn_${BOARD}x${BOARD}k${K}_scratch_seed${SEED}.bin"
TRANSFER="${OUTDIR}/cnn_${BOARD}x${BOARD}k${K}_from3x3_seed${SEED}.bin"

echo "== Transfer A/B =="
echo "seed=${SEED} board=${BOARD} k=${K}"
echo "outdir=${OUTDIR}"
echo

if [[ "${SKIP_3X3}" == "1" ]]; then
  echo "== 1) Using existing 3x3 CNN checkpoint (source) =="
  echo "SKIP_3X3=1, expecting: ${CNN_3X3}"
  [[ -f "${CNN_3X3}" ]] || { echo "Missing init model: ${CNN_3X3}" >&2; exit 1; }
  echo
else
  echo "== 1) Train 3x3 CNN checkpoint (source) =="
  ./target/release/train_alphazero \
    --net-type cnn \
    --board-width 3 --win-k 3 \
    --iterations "${ITERS_3X3}" \
    --seed "${SEED}" \
    --model-path "${CNN_3X3}"
  echo
fi

echo "== 2) Train larger-board CNN from scratch =="
./target/release/train_alphazero \
  --net-type cnn \
  --board-width "${BOARD}" --win-k "${K}" \
  --fixed-suite-every 0 \
  --iterations "${ITERS_BIG}" \
  --seed "${SEED}" \
  --model-path "${SCRATCH}"
echo

echo "== 3) Train larger-board CNN with init-from-3x3 =="
./target/release/train_alphazero \
  --net-type cnn \
  --board-width "${BOARD}" --win-k "${K}" \
  --fixed-suite-every 0 \
  --iterations "${ITERS_BIG}" \
  --seed "${SEED}" \
  --init-model-path "${CNN_3X3}" \
  --model-path "${TRANSFER}"
echo

echo "Done."
