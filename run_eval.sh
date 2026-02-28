#!/bin/bash

# 1. Configuration - Update these paths
MODEL_PATH="./ckpts/ga_train_lt_256/step_00000245"
MODEL_PATH="../ckpts/unlearn_pcgrad_idknpo_feb22_epoch2"
MODEL_PATH="./ckpts/unlearn_pcgrad_ga_feb20_epoch2/step_00000490"
MODEL_PATH="gpt2"               # or path/to/your/checkpoint
MODEL_PATH="../ckpts/ga_train_lt_256_epoch5/step_00002453"
# or path/to/your/checkpoint
DATA_PATH="../data/jan22_test_310.pickle"
BASE_MODEL="gpt2"                  # only needed if MODEL_PATH is a .bin file
OUTPUT_LOG="eval_results.log"

# 2. Virtual Environment (Optional - uncomment if needed)
# source venv/bin/activate

echo "Starting Toxicity Evaluation..."
echo "Model: $MODEL_PATH"
echo "Data:  $DATA_PATH"

# 3. Execution
# We use backslashes (\) to break the command into readable lines
cd eval
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
# cd eval

# TOXIC_DATA="../data/feb9_perpelxity_toxic_1000.pickle"

# echo "gradient ascent"
# MODEL_PATH="../ckpts/ga_train_lt_256/step_00000245"
# python perplexity.py \
#   --data_pickle "$TOXIC_DATA" \
#   --model "$MODEL_PATH" \
#   --base_model gpt2 \
#   --tox_threshold 0.5 \
#   --seq_len 256 --stride 1 \
#   --tqdm

# echo "idknpo"

# MODEL_PATH="../ckpts/idknpo_out_gpt2_feb8_2epoch"
# python perplexity.py \
#   --data_pickle "$TOXIC_DATA" \
#   --model "$MODEL_PATH" \
#   --base_model gpt2 \
#   --tox_threshold 0.5 \
#   --seq_len 256 --stride 1 \
#   --tqdm

  
# echo "gradient ascent pcgrad"

# MODEL_PATH="../ckpts/unlearn_pcgrad_ga_feb20_epoch2/step_00000490"
# python perplexity.py \
#   --data_pickle "$TOXIC_DATA" \
#   --model "$MODEL_PATH" \
#   --base_model gpt2 \
#   --tox_threshold 0.5 \
#   --seq_len 256 --stride 1 \
#   --tqdm

  
# echo "idkNPO pcgrad"

# MODEL_PATH="../ckpts/unlearn_pcgrad_idknpo_feb22_epoch2"
# python perplexity.py \
#   --data_pickle "$TOXIC_DATA" \
#   --model "$MODEL_PATH" \
#   --base_model gpt2 \
#   --tox_threshold 0.5 \
#   --seq_len 256 --stride 1 \
#   --tqdm

# echo "gpt2"

# MODEL_PATH="gpt2"
# python perplexity.py \
#   --data_pickle "$TOXIC_DATA" \
#   --model "$MODEL_PATH" \
#   --base_model gpt2 \
#   --tox_threshold 0.5 \
#   --seq_len 256 --stride 1 \
#   --tqdm

