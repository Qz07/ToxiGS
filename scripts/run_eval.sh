#!/bin/bash

# 1. Configuration - Update these paths
MODEL_PATH="MODEL_PATH_HERE"
# or path/to/your/checkpoint
DATA_PATH="DATA_PATH_HERE"  # e.g., ../data/eval_data.jsonl
BASE_MODEL="gpt2"                  # only needed if MODEL_PATH is a .bin file
OUTPUT_LOG="eval_results.log"

# 2. Virtual Environment (Optional - uncomment if needed)
# source venv/bin/activate

echo "Starting Toxicity Evaluation..."
echo "Model: $MODEL_PATH"
echo "Data:  $DATA_PATH"

# 3. Execution
# We use backslashes (\) to break the command into readable lines
cd ../eval
python3 evaluation.py \
    --model "$MODEL_PATH" \
    --data "$DATA_PATH" \
    --base_model "$BASE_MODEL" \
    --max_new_tokens 64 \
    --temperature 0.8 \
    --top_p 0.95 \
    --do_sample \
    --toxicity_batch_size 32 \
    --score_on "completion" \
    --dtype "auto" | tee "$OUTPUT_LOG"

echo "------------------------------------------------"
echo "Evaluation complete. Results saved to $OUTPUT_LOG"