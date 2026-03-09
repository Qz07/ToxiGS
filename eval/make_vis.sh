# GradDiff vs GA-PCGrad
python pcgrad-non_vis.py \
  --base_model gpt2 \
  --non_ckpt ./ckpts/ga_train_lt_256/step_00000245_hf \
  --pc_ckpt  ./ckpts/unlearn_PCGrad_gradDiff_feb20_epoch2/step_00000490_hf \
  --out_dir  ./results/GA_vs_GA_pcgrad

# If you want idkDPO vs idkdpo-PCGrad (or whatever your intended non/pc pairing is)
python pcgrad-non_vis.py \
  --base_model gpt2 \
  --non_ckpt ./ckpts/idkdpo_out_gpt2_feb8_2epoch_hf \
  --pc_ckpt  ./ckpts/unlearn_pcgrad_idkdpo_feb22_epoch2_hf \
  --out_dir  ./results/idk_vs_pcgrad