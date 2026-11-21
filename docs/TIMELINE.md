# Build timeline

Build log of masterfabric-genai-foundations: transformer, LoRA, GAN and policy-gradient RL written from scratch in PyTorch and trained on an Apple M4 Pro.

Every step below is one commit, in order. Times are Europe/Istanbul (UTC+3).

| # | Commit | Time | Change |
|---|---|---|---|
| 1 | `a3de5b6` | 2026-11-21 05:01 | chore: scaffold repository with AGPL-3.0 license and community files |
| 2 | `d5e16dd` | 2026-11-21 05:04 | feat(core): device selection, seeding and result helpers |
| 3 | `630fe4e` | 2026-11-21 05:04 | feat(transformer): GPT-style decoder written from the attention equations |
| 4 | `78b5280` | 2026-11-21 05:04 | feat(lora): LoRA adapters from scratch with exact merge |
| 5 | `9e8892d` | 2026-11-21 05:04 | test: attention vs torch SDPA, causality, initial loss and LoRA identity/merge |
| 6 | `c2258b7` | 2026-11-21 05:04 | feat(data): training text from MasterFabric repos 1-6 |
| 7 | `502b187` | 2026-11-21 05:04 | feat(gan,rl): toy-ring GAN and REINFORCE on a from-scratch CartPole |
| 8 | `d75474d` | 2026-11-21 05:05 | feat(experiments): pretrain the char-level GPT on MPS with early stopping |
| 9 | `fc5038f` | 2026-11-21 05:06 | fix(lora): create adapter weights on the base layer's device |
| 10 | `7f77604` | 2026-11-21 05:06 | feat(experiments): LoRA vs full fine-tuning, GAN and RL experiment runners |
| 11 | `6dcb765` | 2026-11-21 05:09 | results: LoRA ranks vs full fine-tuning on a 14k-character corpus |
| 12 | `ebfec28` | 2026-11-21 05:09 | results: GAN mode coverage, saturating vs non-saturating loss |
| 13 | `cfbf14a` | 2026-11-21 05:09 | results: REINFORCE on CartPole with and without a baseline |
| 14 | `0fa4fc5` | 2026-11-21 05:10 | docs: charts for GPT pretraining, LoRA, GAN and policy gradient |
| 15 | `a8f5f8b` | 2026-11-21 05:10 | docs(notebooks): four executed notebooks, one per building block |
| 16 | `eb368f7` | 2026-11-21 05:11 | docs: overview diagram of the four from-scratch building blocks |
| 17 | `472918e` | 2026-11-21 05:11 | docs: 1600x900 hero from the measured charts and numbers |
| 18 | `325b167` | 2026-11-21 05:11 | docs(adr): 0001 scope of from scratch |
| 19 | `c598f6f` | 2026-11-21 05:11 | docs(adr): 0002 laptop-scale, reproducible experiments |
| 20 | `b352553` | 2026-11-21 05:11 | docs(adr): 0003 LoRA for adaptation when forgetting matters, and index |
| 21 | `07ca41b` | 2026-11-21 05:11 | build: pinned Python requirements |
| 22 | `595b67c` | 2026-11-21 05:11 | build: Makefile targets for tests, each experiment, notebooks and charts |
| 23 | `3b75eba` | 2026-11-21 05:11 | docs: CONTRIBUTING with from-scratch and measured-results rules |
| 24 | `5d16418` | 2026-11-21 05:12 | docs: full README with results, figures, notebooks and limitations |

The final commit adds this file and the README; see `git log` for the full messages, which explain
the why of each step (including the fixes found by running the system).
