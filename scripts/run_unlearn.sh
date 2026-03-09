#!/usr/bin/env bash
export TOKENIZERS_PARALLELISM=false

MODE="${1:-idkDPO}"


if [[ "$MODE" == "GradDiff" ]]; then
  echo "Running gradient difference unlearning (GradDiff GA)..."
  cd train
  torchrun --nproc_per_node=1 unlearn_graddiff.py \
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
else
  echo "Running IdkDPO unlearning..."
  torchrun --standalone --nproc_per_node=2 train/unlearn_idkdpo.py \
    --data_path "$DATA_PATH" \
    --ckpt_dir "$CKPT_DIR" \
    --ft_filename "$FT_FILE" \
    --output_dir "$OUT_DIR" \
    --bf16 \
    --max_length 256 \
    --batch_size_retain 16 \
    --batch_size_forget 16 \
    --grad_accum 8 \
    --epochs 2 \
    --lr 2e-5 \
    --warmup_steps 50 \
    --beta 0.1 \
    --dpo_coef 1.0 \
    --retain_coef 1.0 \
    --idk_text " I don't know." \
    --idk_lm_coef 0.05 \
    --grad_clip 1.0 \
    --use_no_sync \
    --log_every 10 \
    --save_every 200 \
    --wandb_project "$WANDB_PROJ" \
    --wandb_run_name "$RUN_NAME"
fi