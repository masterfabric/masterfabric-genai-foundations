# MasterFabric GenAI Foundations. Run `make help` for the list of targets.
SHELL := /bin/bash
PY    ?= .venv/bin/python

.PHONY: help venv test gpt lora gan wgan diffusion dpo rl all notebooks charts lab

help: ## Show targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n",$$1,$$2}'

venv: ## Create .venv with uv and the pinned dependencies
	uv venv -p 3.12 .venv && uv pip install -p .venv -r requirements.txt

test: ## Unit tests: attention, LoRA, gradient penalty, diffusion samplers, DPO loss (14)
	$(PY) -m pytest -q

gpt: ## Pretrain the char-level GPT (~1 min on Apple MPS)
	$(PY) experiments/train_gpt.py

lora: gpt ## No tuning vs full fine-tuning vs LoRA r=2/8/32 (~2 min)
	$(PY) experiments/lora_finetune.py

gan: ## GAN, 2 losses x 3 seeds (~30 s, CPU)
	$(PY) experiments/run_gan.py

wgan: ## WGAN-GP at 4k and 20k generator steps, plus the 20k GAN control, 3 seeds each (~8 min, CPU)
	$(PY) experiments/run_wgan_gp.py && $(PY) experiments/run_gan_long.py

diffusion: ## DDPM/DDIM on the ring (3 seeds, CPU) and on 16x16 shapes (U-Net, Apple MPS) (~7 min)
	$(PY) experiments/run_diffusion.py

dpo: gpt ## DPO grid vs filtered SFT and best-of-4 on the pretrained GPT (~3 min, MPS)
	$(PY) experiments/run_dpo.py

rl: ## REINFORCE with and without baseline, 3 seeds each (~25 s, CPU)
	$(PY) experiments/run_rl.py

all: lora gan wgan diffusion dpo rl notebooks charts ## Every experiment, then notebooks and charts

notebooks: ## Rebuild and execute the four notebooks
	$(PY) notebooks/build_notebooks.py

charts: ## Charts, diagram and hero from results/*.json
	cd viz && ../$(PY) make_charts.py && ../$(PY) make_diagrams.py && ../$(PY) make_hero.py

lab: ## Optional: JupyterLab on 127.0.0.1:17888 (needs `uv pip install -p .venv jupyterlab`)
	$(PY) -m jupyter lab --ip 127.0.0.1 --port 17888 --no-browser notebooks/
