# 0004. WGAN-GP and diffusion on the same 8-Gaussian ring, with a longer-training control

- Status: Accepted
- Date: 2026-11-21

## Context

The GAN of section 3 covers 4 to 6 of the 8 modes. To show what fixes mode collapse, new models must be scored on
the same target with the same metric (modes that receive at least 2% of samples within 3 standard deviations, and the
share of samples that land on a mode), on the same 3 seeds.

## Decision

- WGAN-GP (`gaf/gan.py: train_wgan_gp`) uses the same generator and the same critic network (no sigmoid) with the
  paper's settings: 5 critic steps per generator step, gradient-penalty weight 10, Adam(1e-4, betas 0.5/0.9).
- It is reported at the GAN's budget (4,000 generator steps) and at 20,000, because at 4,000 its curves are still
  rising. The non-saturating GAN is also trained for 20,000 steps, so the comparison is not just more training.

## Evidence (`results/wgan_gp.json`, `results/gan_long.json`, `results/gan.json`)

| Model, generator steps | Modes covered (seeds 0/1/2) | % of samples on a mode |
|---|---|---|
| GAN non-saturating, 4k | 5 / 4 / 6 | 75 / 78 / 76 |
| GAN non-saturating, 20k | 5 / 4 / 6 | 88 / 91 / 88 |
| WGAN-GP, 4k | 4 / 2 / 0 | 20 / 11 / 4 |
| WGAN-GP, 20k | **8 / 8 / 8** | 76 / 77 / 74 |
| Diffusion, DDPM 1000 sampling steps (ADR 0005) | **8 / 8 / 8** | 95 / 95 / 95 |

## Consequences

- More GAN training sharpens the modes it already has (on-mode share 76% → 89%) but finds no new ones; WGAN-GP
  finds all eight, at lower sharpness and about 4x the GAN's wall time per run (101-107 s vs 26-32 s per seed on
  the CPU; timings measured while other experiments ran in parallel).
- At equal generator steps WGAN-GP is worse. "Which GAN is better" depends on the budget.
- Three seeds on a 2-D toy. The ranking is clear here; it says nothing about image-scale GANs.
