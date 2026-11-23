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


## Addition: WGAN-GP, a small diffusion model and DPO (2026-11-21)

Started from `bfc85b3`. Times are Europe/Istanbul. No new dependencies: everything uses the pinned torch, numpy and
matplotlib.

| # | Commit | Time | Change |
|---|---|---|---|
| 1 | `bffa1e8` | 2026-11-21 13:48 | gan: WGAN-GP critic and gradient penalty from the paper's equations, with tests |
| 2 | `89218e3` | 2026-11-21 13:48 | results: WGAN-GP on the 8-Gaussian ring, 3 seeds at 4,000 and 20,000 generator steps (8/8 modes at 20k) |
| 3 | `cc08dc5` | 2026-11-21 13:48 | results: control, non-saturating GAN at 20,000 steps still covers 4-6 of 8 modes |
| 4 | `4874e15` | 2026-11-21 13:49 | diffusion: DDPM training and sampling, DDIM with optional x0 clamp, point and U-Net denoisers, synthetic shapes and a judge classifier |
| 5 | `9c416ba` | 2026-11-21 13:49 | dpo: DPO loss, completion log-probs and a programmatic word-validity reward, with tests |
| 6 | `1f733a6` | 2026-11-21 13:56 | results: diffusion, ring 8/8 modes with DDPM and clamped DDIM; 16x16 shapes, classifier-feature Frechet 0.75 (DDPM) vs collapse without the clamp |
| 7 | `d68deaf` | 2026-11-21 14:05 | results: DPO grid (3 learning rates x beta 0.1/0.3/1.0) vs filtered SFT and best-of-4; small beta pushes chosen log-probs down too |
| 8 | `890c6b0` | 2026-11-21 14:06 | charts: ring comparison (GAN, WGAN-GP, diffusion), diffusion shapes and step counts, DPO reward vs KL |
| 9 | `c4ae784` | 2026-11-21 14:07 | diagram: overview with WGAN-GP, diffusion and DPO |
| 10 | `90265fe` | 2026-11-21 14:07 | hero: diffusion, DPO and the ring comparison, read from results |
| 11 | `b504aa9` | 2026-11-21 14:07 | Makefile: wgan, diffusion and dpo targets |
| 12 | `8f5a896` | 2026-11-21 14:08 | docs: ADRs 0004-0006 (WGAN-GP on the ring, diffusion sampling, DPO) |
| 13 | `03458fb` | 2026-11-21 14:10 | README: WGAN-GP, diffusion and DPO sections with figures 6-9, key results, limitations and roadmap |
| 14 | (this commit) | | docs(timeline): this section |

### Verification (measured on the M4 Pro)

| Check | Result |
|---|---|
| Unit tests | 14 passed (gradient penalty by hand, DDPM/DDIM recover a point with an exact denoiser, DPO loss and log-probs) |
| GAN control, 20k steps | 5 / 4 / 6 of 8 modes, 88-91% on a mode |
| WGAN-GP, 4k / 20k generator steps | 4 / 2 / 0 and 8 / 8 / 8 modes; 20k: 74-77% on a mode, 101-107 s per seed |
| Diffusion, ring | DDPM 8/8 modes, 94.9% on a mode; DDIM without clamp 2.3 modes; DDIM 20 steps + clamp 8/8, 89.2% |
| Diffusion, 16x16 shapes | U-Net 582k params, 8,000 steps in 182 s on MPS; judge accuracy 1.000; Fréchet 0.75 (DDPM) vs 0.80 real vs real; DDIM 10 + clamp 2.68 |
| DPO | 1,722 pairs; reward 0.749 → 0.786 ± 0.006 (β 1.0, lr 1e-5); filtered SFT best 0.771; best-of-4 0.763 |
| Machine state | experiments ran as local Python processes only, all finished; no servers, Colima or k3d used; ports 3000, 8081, 11234 untouched |

### Found while running it

- WGAN-GP with the paper's settings is slower to learn than the GAN: at 4,000 generator steps it covers 0-4 modes.
  The 20,000-step control shows the GAN does not catch up with more steps; it only sharpens the modes it has.
- DDIM failed on both tasks until the predicted x0 was clamped: the cosine schedule's last ᾱ is 2.4e-9, so the
  first DDIM step divides the network error by about 5e-5. On images the unclamped sampler gave noise that the
  judge called "cross" 100% of the time.
- The first shapes run produced crosses only: the weight EMA (fixed 0.999) still held 37% of the random initial
  weights at step 1,000. The EMA now warms up.
- The DDPM samples' Fréchet distance (0.75) is below the real-vs-real value (0.80): the judge is at its noise floor
  and cannot rank DDPM against real data, only samplers against each other.
- The first DPO grid (lr 1e-4, β 0.03-0.3) lowered the reward it was trained on (0.35-0.65 from 0.75) and pushed
  validation loss to 2.1-4.0. Small β lowers the probability of the chosen completions as well as the rejected ones.
- MPS is not bit-deterministic: re-running the DPO grid moved one point (β 1.0, lr 1e-4) from 0.729 to 0.719.
- WGAN-GP, the GAN control and the diffusion ring runs overlapped on the CPU, so their wall times are upper bounds.

## Addition: classifier-free guidance, iterative DPO and a measurement bug (2026-11-23)

Times are Europe/Istanbul.

| Commit | Time | What |
|---|---|---|
| `265b5a4` | 2026-11-23 12:44 | diffusion: class-conditional U-Net with a null class and classifier-free guidance; unit test for the guidance combination |
| `b9191a9` | 2026-11-23 12:44 | experiments: classifier-free guidance sweep (w = 0..8) on the 16x16 shapes with the real-shape judge |
| `78f6e08` | 2026-11-23 12:44 | experiments: DPO with beta chosen on selection prompts, reported on unseen test prompts, plus an on-policy second round |
| `85f58cf` | 2026-11-23 13:27 | fix: evaluate() restores the caller's train/eval mode; it switched dropout on in DPO's reference model and samplers after the first measurement (regression test) |
| `d9d54dc` | 2026-11-23 13:27 | results: keep the DPO and iterative-DPO runs measured with the dropout bug, for comparison |
| `667dd6e` | 2026-11-23 13:34 | results: DPO grid and iterative DPO re-measured in eval mode; classifier-free guidance sweep (w = 0..8) |
| `7ddf1ec` | 2026-11-23 13:34 | charts: guidance samples and distance vs w (figure 10); DPO chart and hero from the corrected results; make cfg, make dpo-iter |
| `fdca02d` | 2026-11-23 13:34 | docs: section 8 classifier-free guidance, corrected DPO table with the dropout bug explained, iterative DPO; ADR 0007, ADR 0006 update |
| (this commit) | 2026-11-23 | docs(timeline): guidance, iterative DPO and the eval-mode fix |

### Verification

| Check | Result |
|---|---|
| Unit tests | 16 passed (new: guidance combination, `evaluate()` keeps the caller's mode; the second fails on the old code) |
| Guidance sweep (`make cfg`) | 8,000 steps in 187 s on MPS; w = 0 / 1 / 8: requested class 23.2% / 99.8% / 100%, Fréchet 1.93 / 0.57 / 41.3 (real vs real 0.80) |
| DPO grid re-run (`make dpo`) | pairs identical to the first run (1,722; chosen 0.848, rejected 0.608); DPO β=1.0 0.783 at KL 1.2; best-of-4 0.918 |
| Iterative DPO (`make dpo-iter`) | β=1.0 selected on 128 prompts; 256 test prompts: 0.763 → 0.792 → 0.822 |
| Other experiments | not re-run; their charts regenerate unchanged from the existing results |

### Found while running it

- `evaluate()` ended with `model.train()`, so the DPO reference model ran with dropout after its first
  measurement. The first DPO grid's KL values were 5-10x too high and best-of-4 was 0.763 instead of 0.918, which
  reversed the comparison with DPO. It showed up because the iterative run measured the same reference twice: 0.734
  on selection prompts, then 0.562 on test prompts. Re-generating the round-1 pairs in a clean process reproduced the
  first run's pairs exactly (1,722, 0.8476 / 0.6076), which pointed at state left behind by an earlier call.
- Guidance above w = 1 only moved samples away from real shapes here: the conditional model already draws the
  requested class 99.8% of the time, so the class-match gain has no room to show.
- Circles drift most under guidance (Fréchet 1.4 at w = 1, 157 at w = 8); crosses barely (0.9 to 1.6).
- A background job on this laptop removed every `.venv` under `~/Documents/GitHub` during the work; runs used a
  temporary venv outside the repository with the pinned torch 2.15.0.
