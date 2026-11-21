# MasterFabric GenAI Foundations. Run `make help` for the list of targets.
SHELL := /bin/bash
PY    ?= .venv/bin/python

.PHONY: help venv test gpt lora gan rl all notebooks charts lab

help: ## Show targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n",$$1,$$2}'

venv: ## Create .venv with uv and the pinned dependencies
	uv venv -p 3.12 .venv && uv pip install -p .venv -r requirements.txt

test: ## Unit tests: attention vs torch SDPA, causality, initial loss, LoRA identity and merge
	$(PY) -m pytest -q

gpt: ## Pretrain the char-level GPT (~1 min on Apple MPS)
	$(PY) experiments/train_gpt.py

lora: gpt ## No tuning vs full fine-tuning vs LoRA r=2/8/32 (~2 min)
	$(PY) experiments/lora_finetune.py

gan: ## GAN, 2 losses x 3 seeds (~30 s, CPU)
	$(PY) experiments/run_gan.py

rl: ## REINFORCE with and without baseline, 3 seeds each (~25 s, CPU)
	$(PY) experiments/run_rl.py

all: lora gan rl notebooks charts ## Every experiment, then notebooks and charts

notebooks: ## Rebuild and execute the four notebooks
	$(PY) notebooks/build_notebooks.py

charts: ## Charts, diagram and hero from results/*.json
	cd viz && ../$(PY) make_charts.py && ../$(PY) make_diagrams.py && ../$(PY) make_hero.py

lab: ## Optional: JupyterLab on 127.0.0.1:17888 (needs `uv pip install -p .venv jupyterlab`)
	$(PY) -m jupyter lab --ip 127.0.0.1 --port 17888 --no-browser notebooks/
