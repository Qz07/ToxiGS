ToxiTIGS
========

This repository contains code for training and unlearning toxicity in GPT‑2 style language models.
It includes scripts for:

- Baseline next‑token training on a prompt / generation dataset
- Gradient‑ascent unlearning of toxic generations
- "I don't know" DPO‑style unlearning (IdkDPO)
- PCGrad variants of the above objectives
- Evaluation via toxicity scoring, perplexity, and MMLU


Repository layout
-----------------

- train/ – training and unlearning code
	- gpt2-train.py – FSDP next‑token pretraining / finetuning on a pickle dataset
	- unlearn_graddiff.py – gradient‑ascent unlearning on label=1 examples (optionally mixed with retain loss)
	- unlearn_idkdpo.py – IdkDPO unlearning that pushes toxic generations toward an "I don't know" response
	- PCGrad_gradDiff.py, PCGrad_idkdpo.py – PCGrad versions of GradDiff and IdkDPO unlearning
- eval/ – evaluation utilities
	- inference_utils.py – shared inference helpers
	- evaluation.py – generate completions and score toxicity with a classifier
	- perplexity.py – perplexity evaluation on toxic / non‑toxic sets
	- run_mmlu.py – MMLU evaluation for a given checkpoint
- scripts/ – convenience shell scripts
	- run_train.sh – baseline GPT‑2 training
	- run_unlearn.sh – example GradDiff / IdkDPO unlearning runs
	- run_pcgrad.sh – PCGrad IdkDPO unlearning
	- run_eval.sh – toxicity + perplexity evaluation
	- run_mmlu.sh – MMLU evaluation wrapper
- notebooks/ – exploratory notebooks for data prep, inference, and analysis


Data format
-----------

Most training and unlearning scripts expect a pickle file containing either a list or dict of
examples with the following keys:

- prompt: input prompt string
- generation: model generation string
- label: int, typically 1 for toxic / forget and 0 for retain

For toxicity evaluation, eval/evaluation.py expects a pickle file with a list of dicts containing:

- text: prompt string to feed into the model


Environment and dependencies
----------------------------

- Python 3.10+
- PyTorch with CUDA (FSDP requires GPUs)
- transformers, peft, datasets‑style tooling
- wandb (optional, can be disabled if not installed)

A simple way to set up an environment is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or CPU wheel
pip install transformers peft tqdm wandb numpy
```


Baseline training
-----------------

To train a GPT‑2 baseline on your dataset (example from scripts/run_train.sh):

```bash
cd train
torchrun --nproc_per_node=2 gpt2-train.py \
	--data_path ../data/jan26_filter_lt_256_248k.pickle \
	--output_dir ../ckpts/train_lt_256 \
	--model_name gpt2 \
	--seq_len 256 \
	--epochs 1 \
	--batch_size 32 \
	--grad_accum 8 \
	--lr 2e-4 \
	--use_wandb --wandb_project gpt2-next-token --run_name train_lt_256
```

Edit the paths, model size, and hyperparameters to match your setup.


Unlearning methods
------------------

Gradient‑ascent unlearning (GA):

```bash
cd train
torchrun --nproc_per_node=1 unlearn_graddiff.py \
	--data_path ../data/jan26_filter_lt_256_248k.pickle \
	--model_name_or_path ../ckpts/train_lt_256/step_00000484 \
	--base_model gpt2 \
	--output_dir ../ckpts/ga_train_lt_256_epoch5 \
	--epochs 5 --batch_size 32 --grad_accum 8 --seq_len 256 \
	--lr 2e-5 --forget_weight 1.0 --retain_weight 1.0
```

IdkDPO + PCGrad:

```bash
torchrun --standalone --nproc_per_node=2 PCGrad_idkdpo.py \
	--data_path ./data/jan26_filter_lt_256_248k.pickle \
	--ckpt_dir ./ckpts/train_lt_256/step_00000484 \
	--ft_filename pytorch_model.bin \
	--base_model gpt2 \
	--output_dir ./ckpts/unlearn_pcgrad_idkdpo_feb22_epoch2 \
	--bf16 --max_length 256 --batch_size_retain 16 --batch_size_forget 16 \
	--grad_accum 8 --epochs 2 --lr 2e-5 --warmup_steps 50 \
	--beta 0.1 --dpo_coef 1.0 --retain_coef 1.0
```


Evaluation
----------

Toxicity evaluation (scripts/run_eval.sh):

```bash
cd scripts
bash run_eval.sh
```

This script calls eval/evaluation.py to generate completions for prompts in a pickle file and scores
them with a toxicity classifier (e.g., unitary/unbiased-toxic-roberta), printing summary statistics
and saving logs.

MMLU evaluation (scripts/run_mmlu.sh):

```bash
cd scripts
bash run_mmlu.sh
```

Results are written to JSON files under results/.

License
-------

See LICENSE for licensing details.
