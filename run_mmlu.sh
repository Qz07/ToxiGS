#!/usr/bin/env bash
set -euo pipefail

# --------- user config ----------
CKPT="./ckpts/ga_train_lt_256_epoch5/step_00002453_hf"          # <-- change this
OUT_DIR="./results"
OUT_JSON="${OUT_DIR}/mmlu_gpt2.json"
LOG_FILE="${OUT_DIR}/mmlu_gpt2.log"

# Single A5000 settings
DEVICE="cuda:0"
DTYPE="float16"
BATCH_SIZE="auto"
NUM_FEWSHOT=0

# Optional: for a quick smoke test set LIMIT=50 (or comment out)
LIMIT=""   # e.g., "50" or "0.1" ; leave empty for full eval
# --------------------------------

mkdir -p "${OUT_DIR}"

CMD=(python -u run_mmlu.py
  --ckpt "${CKPT}"
  --out "${OUT_JSON}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --batch_size "${BATCH_SIZE}"
  --num_fewshot "${NUM_FEWSHOT}"
)

if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi

echo "Running command:"
printf ' %q' "${CMD[@]}"
echo
echo "Logging to: ${LOG_FILE}"

# Run + tee logs
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"

echo "Done."
echo "Results: ${OUT_JSON}"