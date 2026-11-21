<p align="center"><img src="docs/hero.png" alt="MasterFabric GenAI Foundations: transformer, LoRA, GAN and RL from scratch with measured results" width="100%"></p>

# MasterFabric · GenAI Foundations

The building blocks of generative AI, written from their equations in PyTorch and trained on a
laptop GPU (Apple M4 Pro, MPS):

1. **Attention and a GPT-style transformer**: a character-level language model.
2. **LoRA**: low-rank adapters, compared with full fine-tuning, including how much each forgets.
3. **A GAN**: generator and discriminator on a target where mode collapse can be counted.
4. **Policy-gradient RL**: REINFORCE with and without a value baseline on a from-scratch CartPole.

Each comes with tests, an executed notebook and results in [`results/`](results/). All numbers are
measured, including the runs that overfit or collapsed.

> Part of the MasterFabric data and AI portfolio. The text the models learn from is this
> portfolio's own documentation.

## Contents

[Key results](#key-results) · [Overview](#overview) · [1 Transformer](#1-attention-and-a-gpt-style-transformer) ·
[2 LoRA](#2-lora) · [3 GAN](#3-gan) · [4 RL](#4-policy-gradient-rl) · [Notebooks](#notebooks) · [ADRs](#decisions) ·
[Enterprise mapping](#enterprise-mapping) · [Quickstart](#quickstart) · [Limitations](#limitations) · [Roadmap](#roadmap)

## Key results

| Block | Result | Time (M4 Pro) |
|---|---|---|
| Attention | matches `torch.nn.functional.scaled_dot_product_attention` within 1.5e-7 | test |
| GPT, 0.82M params | validation loss 4.72 (uniform guess) → **1.516** nats/char | 48 s on MPS, 257,929 tok/s |
| LoRA r=8, 9% of weights | new-corpus loss **1.758** vs 1.725 for full fine-tuning; original-corpus loss **1.642** vs 1.747 (less forgetting) | 21 s |
| GAN, non-saturating loss | **4-6 of 8** modes covered, 75-78% of samples on a mode (3 seeds) | 5 s per run, CPU |
| REINFORCE on CartPole | solved (mean return ≥ 195) after **352-560** episodes in all 6 runs | 4 s per run, CPU |

## Overview

<p align="center"><img src="docs/images/overview.png" alt="The four from-scratch building blocks" width="100%"></p>
<p align="center"><sub><b>Figure 1.</b> What is hand-written. Only tensors, autograd, basic layers and optimisers come from PyTorch (<a href="docs/adr/0001-from-scratch-scope.md">ADR 0001</a>).</sub></p>

```mermaid
flowchart LR
  A[(repos 1-3 docs<br/>104k chars)] -->|pretrain| G[GPT<br/>attention from scratch]
  G -->|frozen base| L[LoRA adapters<br/>r = 2 · 8 · 32]
  B[(repos 4-6 ADRs<br/>14k chars)] -->|adapt| L
  G -.->|compare| F[full fine-tuning]
  B --> F
  N[noise z] --> GEN[generator] --> D{discriminator}
  R[(8-Gaussian ring)] --> D
  E[CartPole in torch] <--> P[policy π + value V]
```

## 1. Attention and a GPT-style transformer

[`gaf/transformer.py`](gaf/transformer.py) implements `softmax(QKᵀ/√d + causal mask) · V`, a
multi-head split, pre-norm residual blocks with a GELU MLP, learned positions, weight tying and
top-k sampling. Tests check:
- the attention against PyTorch's fused kernel;
- that changing future tokens never changes past logits;
- that an untrained model starts at the uniform loss.

<p align="center"><img src="docs/images/gpt_loss.png" alt="GPT training and validation loss" width="85%"></p>
<p align="center"><sub><b>Figure 2.</b> 4 layers, 4 heads, d_model 128, context 128, dropout 0.2, 1,500 steps of 64 × 128 characters. The train/validation gap shows the corpus is small.</sub></p>

| | |
|---|---|
| Parameters | 824,064 (vocabulary of 112 characters) |
| Data | 103,516 training / 11,502 validation characters |
| Best validation loss | 1.516 nats/char (uniform guess: 4.72) |
| Throughput | 257,929 tokens/s training on Apple MPS (48 s) |
| First attempt | d_model 192, dropout 0.1: overfit by step 700 (train 0.72, validation rising to 1.75). Replaced, and kept in the commit history |

A sample after training (temperature 0.8) has the structure of the documents: Markdown headings,
image tags, words like "telemetry", "watermarks" and "Spark". The sentences do not make sense. That
is what 0.8M parameters and 100 KB of text can do.

## 2. LoRA

[`gaf/lora.py`](gaf/lora.py) computes `W x + (α/r) · B A x`. W is frozen and B starts at zero, so
training starts exactly from the base model. `merged()` folds the adapter back into one matrix, so
inference has no extra cost. The pretrained GPT is adapted to a different corpus: the ADRs of repos 4-6.

<p align="center"><img src="docs/images/lora.png" alt="LoRA vs full fine-tuning: target loss, forgetting, learning curves" width="100%"></p>
<p align="center"><sub><b>Figure 3.</b> Left: validation loss on the new corpus (solid) and on the original corpus (hatched, forgetting), at the best checkpoint. Right: learning curves. Every method overfits the 14k-character corpus after 50-300 steps.</sub></p>

| Method | Trainable | Val loss, new corpus | Val loss, original | Best step | Time |
|---|---|---|---|---|---|
| No fine-tuning | 0 | 2.193 | 1.516 | | |
| Full fine-tuning | 824,064 (100%) | **1.725** | 1.747 | 100 | 14.9 s |
| LoRA r=2 | 18,432 (2.2%) | 1.754 | 1.718 | 300 | 19.7 s |
| LoRA r=8 | 73,728 (9.0%) | 1.758 | **1.642** | 50 | 20.9 s |
| LoRA r=32 | 294,912 (35.8%) | 1.738 | 1.825 | 50 | 19.9 s |

LoRA gets within 0.03 of full fine-tuning while training 2-9% of the weights. At r=2 and r=8 it also
forgets less. It is *not* faster here: on a model this small, the extra matmuls cost more than the
smaller optimiser state saves ([ADR 0003](docs/adr/0003-lora-for-adaptation.md)).

## 3. GAN

[`gaf/gan.py`](gaf/gan.py) trains two 3-layer MLPs adversarially on a ring of eight 2-D Gaussians.
The target is simple on purpose: it makes **mode collapse countable**. A mode counts as covered when
it receives at least 2% of the samples, and a sample is on-mode within 3σ of a mode centre.

<p align="center"><img src="docs/images/gan.png" alt="GAN samples over training and modes covered" width="100%"></p>
<p align="center"><sub><b>Figure 4.</b> Samples from one run at steps 0, 500, 1,500 and 4,000, and modes covered for 3 seeds per loss.</sub></p>

| Generator loss | Modes covered (of 8) | Samples on a mode | Seeds |
|---|---|---|---|
| Non-saturating, −log D(G(z)) | 4, 5, 6 | 75.1-77.5% | 0, 1, 2 |
| Saturating, log(1 − D(G(z))) (original minimax) | 5, 6, 5 | 32.1-61.2% | 0, 1, 2 |

The non-saturating loss gives sharper samples. Neither loss recovers all 8 modes in 4,000 steps:
the GAN lands its samples near the ring but drops some modes. That partial mode collapse is the
measured result, not a tuned-away one.

## 4. Policy-gradient RL

[`gaf/rl.py`](gaf/rl.py) reimplements the cart-pole equations of motion (Barto, Sutton and Anderson,
1983) in torch, vectorised over 16 parallel episodes capped at 200 steps. REINFORCE uses normalised
discounted returns, optionally minus a learned value baseline.

<p align="center"><img src="docs/images/rl.png" alt="CartPole returns per seed with and without baseline" width="85%"></p>
<p align="center"><sub><b>Figure 5.</b> Mean episode return per update for 3 seeds of each variant. One plain-REINFORCE run collapses to 150 after about 1,100 episodes and recovers.</sub></p>

| Variant | Episodes until mean return ≥ 195 | Final return (last 10 updates) | Worst return after 600 episodes |
|---|---|---|---|
| REINFORCE | 400, 352, 560 | 189.5, 198.8, 198.8 | 149.5, 188.1, 180.4 |
| + value baseline | 480, 368, 496 | 198.6, 198.7, 200.0 | 182.3, 181.9, 181.6 |

At this scale, the baseline does not speed up learning. It does make the policy more stable after
it is solved (worst dip 182 vs 150).

## Notebooks

Generated and executed by [`notebooks/build_notebooks.py`](notebooks/build_notebooks.py), so every
output is real and the cells can be reviewed as Python:

| Notebook | Live in the notebook |
|---|---|
| [01_attention_and_gpt](notebooks/01_attention_and_gpt.ipynb) | attention vs SDPA (1.5e-7), the causal attention matrix, text generated by the trained model |
| [02_lora](notebooks/02_lora.ipynb) | identity at init, exact merge, parameter counts, the comparison table |
| [03_gan](notebooks/03_gan.ipynb) | a 1,500-step GAN run with its scatter plot |
| [04_policy_gradient](notebooks/04_policy_gradient.ipynb) | a 40-update REINFORCE run reaching ~199 |

## Decisions

| ADR | Decision |
|---|---|
| [0001](docs/adr/0001-from-scratch-scope.md) | What "from scratch" means here |
| [0002](docs/adr/0002-laptop-scale-experiments.md) | Laptop-scale experiments with measured, reproducible results |
| [0003](docs/adr/0003-lora-for-adaptation.md) | Use LoRA for domain adaptation when forgetting matters |

## Enterprise mapping

| This repo | Production equivalent |
|---|---|
| `gaf/transformer.py` | Llama, Qwen and Mistral architectures (RoPE, GQA, SwiGLU on top of the same blocks); FlashAttention for the fused kernel |
| `gaf/lora.py` | Hugging Face PEFT, QLoRA; LoRA serving in vLLM (multi-adapter); Cloudera AI fine-tuning studio |
| Forgetting measured on the original corpus | Regression evaluation on general benchmarks before shipping an adapter |
| `gaf/gan.py` | Diffusion models for images today; GANs still used for synthetic tabular data (CTGAN) and super-resolution |
| `gaf/rl.py` | PPO and GRPO in RLHF pipelines (TRL, OpenRLHF), where the policy is the LLM and the reward a learned model |
| Executed notebooks + `results/*.json` | Experiment tracking (MLflow, see masterfabric-airflow-mlops) |

## Quickstart

```bash
git clone https://github.com/masterfabric/masterfabric-genai-foundations && cd masterfabric-genai-foundations
make venv        # uv, Python 3.12, pinned torch / notebook / plotting versions
make test        # 5 tests
make all         # GPT, LoRA, GAN, RL (about 4 minutes on an M4 Pro), then notebooks and charts
```

It runs on CUDA or CPU as well. `gaf.common.device()` picks MPS, then CUDA, then CPU. To browse
interactively, run `make lab` (JupyterLab on `127.0.0.1:17888`, after `uv pip install -p .venv jupyterlab`).

## Limitations

- **Toy scale on purpose.** 0.8M parameters, a 104k-character corpus, a 2-D GAN and CartPole. The
  mechanics are those of real systems; the numbers do not transfer to real model sizes.
- **Small data, so overfitting.** The GPT has a train/validation gap of 0.5 nats. Every fine-tuning
  method overfits the 14k-character corpus within 50-300 steps; results use the best validation
  checkpoint.
- **LoRA speed and memory benefits are not shown.** They appear at billion-parameter scale; here LoRA
  was slower in wall time.
- **Few seeds.** GAN and RL use 3 seeds per setting. GPT and LoRA use one seed. Differences smaller
  than the seed-to-seed spread should not be read as real.
- **Simplified transformer.** Learned absolute positions, standard multi-head attention and GELU. No
  RoPE, GQA, SwiGLU or KV cache in generation.
- **No diffusion model.** GANs are shown because they were named in the brief. Diffusion is on the roadmap.
- **Shared machine.** Timings are from a laptop running other work and were not repeated.

## Roadmap

- KV cache in `generate()` and a measured speed-up; RoPE and grouped-query attention.
- A small DDPM (diffusion) on the same 2-D target, compared with the GAN on mode coverage.
- PPO and a toy RLHF loop: a learned reward model scoring GPT samples.
- QLoRA-style 4-bit base weights with LoRA on top.

## License

AGPL-3.0, see [LICENSE](LICENSE). The additional attribution and trademark terms, and the
third-party licenses, are listed in [NOTICE](NOTICE).
See also [CONTRIBUTING](CONTRIBUTING.md), [CODE_OF_CONDUCT](CODE_OF_CONDUCT.md) and [SECURITY](SECURITY.md).
