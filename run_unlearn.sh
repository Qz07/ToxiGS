#!/usr/bin/env bash
export TOKENIZERS_PARALLELISM=false

DATA_PATH="./data/jan26_filter_lt_256_248k.pickle"
CKPT_DIR="./ckpts/train_lt_256/step_00000484"   # contains config.json + model.safetensors etc
OUT_DIR="./ckpts/feb27_ga_epoch5"
WANDB_PROJ="ToxicGS-unlearning"
RUN_NAME="feb27_ga_epoch5"
FT_FILE="pytorch_model.bin"


# torchrun --standalone --nproc_per_node=2 unlearn_idknpo.py \
#   --data_path "$DATA_PATH" \
#   --ckpt_dir "$CKPT_DIR" \
#   --ft_filename "$FT_FILE" \
#   --output_dir "$OUT_DIR" \
#   --bf16 \
#   --max_length 256 \
#   --batch_size_retain 16 \
#   --batch_size_forget 16 \
#   --grad_accum 8 \
#   --epochs 2 \
#   --lr 2e-5 \
#   --warmup_steps 50 \
#   --beta 0.1 \
#   --dpo_coef 1.0 \
#   --retain_coef 1.0 \
#   --idk_text " I don't know." \
#   --idk_lm_coef 0.05 \
#   --grad_clip 1.0 \
#   --use_no_sync \
#   --log_every 10 \
#   --save_every 200 \
#   --wandb_project "$WANDB_PROJ" \
#   --wandb_run_name "$RUN_NAME"







# torchrun --nproc_per_node=2 unlearn_npo.py \
#   --data_path "$DATA_PATH" \
#   --ckpt_dir "$CKPT_DIR" \
#   --output_dir "$OUT_DIR" \
#   --bf16 \
#   --beta 0.1 \
#   --lm_coef 0.1 \
#   --use_no_sync

cd train
torchrun --nproc_per_node=1 unlearn_ga.py \
  --data_path ../data/jan26_filter_lt_256_248k.pickle \
  --model_name_or_path ../ckpts/train_lt_256/step_00000484 \
  --base_model gpt2 \
  --output_dir ../ckpts/ga_train_lt_256_epoch5 \
  --epochs 5 --batch_size 32 --grad_accum 8 --seq_len 256 \
  --lr 2e-5 \
  --forget_weight 1.0 \
  --retain_weight 1.0 \
  --wandb_project ToxicGS-unlearning \
  --run_name gpt2-ga-unlearn_retain_feb27



# #!/usr/bin/env bash
# set -euo pipefail


# export TOKENIZERS_PARALLELISM=false

# torchrun --nproc_per_node=2 unlearn_rmu.py \
#   --data_path "$DATA_PATH" \
#   --ckpt_dir  "$CKPT_DIR" \
#   --output_dir "$OUT_DIR" \
#   --seq_len 256 --batch_size 32 --grad_accum 8 \
#   --epochs 1 --lr 2e-5 \
#   --alpha 4.0 --c 1.0 --rmu_layer 8 \
#   --fp16 \
#   --wandb_project "$WANDB_PROJ" \
#   --wandb_run_name "$RUN_NAME"
