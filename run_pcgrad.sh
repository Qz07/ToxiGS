# torchrun --nproc_per_node=2 PCGrad_GA.py \
#   --data_path ./data/jan26_filter_lt_256_248k.pickle \
#   --model_name_or_path ./ckpts/train_lt_256/step_00000484 \
#   --base_model gpt2 \
#   --output_dir ./ckpts/unlearn_pcgrad_ga_feb20_epoch2 \
#   --seq_len 256 \
#   --batch_size 32 \
#   --grad_accum 8 \
#   --epochs 2 \
#   --lr 2e-5 \
#   --forget_weight 1.0 \
#   --retain_weight 1.0 \
#   --scheduler cosine \
#   --warmup_steps 50 \
#   --save_every 250 \
#   --log_every 20 \
#   --wandb_project gpt2-unlearning-pcgrad \
#   --run_name pcgrad_ga_gd_epoch2

torchrun --standalone --nproc_per_node=2 PCGrad_idknpo.py \
  --data_path ./data/jan26_filter_lt_256_248k.pickle \
  --ckpt_dir ./ckpts/train_lt_256/step_00000484 \
  --ft_filename pytorch_model.bin \
  --base_model gpt2 \
  --output_dir ./ckpts/unlearn_pcgrad_idknpo_feb22_epoch2 \
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
  --wandb_project gpt2-unlearning-pcgrad \
  --wandb_run_name "gpt2-idkdpo-pcgrad_epoch2_feb22"