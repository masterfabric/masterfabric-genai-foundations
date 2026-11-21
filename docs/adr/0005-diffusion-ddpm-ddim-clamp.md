# 0005. Diffusion: DDPM with a cosine schedule; DDIM only with a clamped x0 prediction

- Status: Accepted
- Date: 2026-11-21

## Context

A diffusion model was added from the equations (`gaf/diffusion.py`): forward noising, an ε-prediction network trained
with MSE, ancestral DDPM sampling over all T = 1,000 steps, and deterministic DDIM sampling on fewer steps. Two
tasks: the ring of ADR 0004 (MLP denoiser, 6,000 steps, CPU) and 16x16 synthetic shapes (circle, square, triangle,
cross; U-Net with 582k parameters, 8,000 steps of fresh images, Apple MPS, 182 s). Generated images are judged by
a CNN trained on real shapes only (test accuracy 1.000): class shares, and a Fréchet distance between Gaussians fitted
to the judge's 128-dimensional features (the FID formula with our own features, not Inception).

## Decision

- Cosine noise schedule (Nichol and Dhariwal, 2021).
- DDIM always clamps the predicted x0 to the data range ([-1, 1] for images, [-2.5, 2.5] for the ring) and recomputes ε
  from it. Without the clamp, DDIM fails.

## Evidence (`results/diffusion.json`)

| Sampler | Ring: modes / on-mode (mean of 3 seeds) | Shapes: Fréchet (judge) | Shapes: class shares | Time for 2,000 images |
|---|---|---|---|---|
| DDPM, 1,000 steps | 8 / 94.9% | **0.75** | 27 / 24 / 26 / 24% | 146 s |
| DDIM, 50 steps, no clamp | 2.3 / 13.6% | 4,726 | 100% "cross" (noise) | 7.0 s |
| DDIM, 100 steps + clamp | 8 / 91.6% | 1.10 | balanced | 14.0 s |
| DDIM, 50 steps + clamp | 8 / 91.4% | 1.24 | balanced | 7.0 s |
| DDIM, 20 steps + clamp | 8 / 89.2% | 1.67 | balanced | 2.8 s |
| DDIM, 10 steps + clamp | 8 / 80.6% | 2.68 | balanced | 1.4 s |
| Real vs real (noise floor) | | 0.80 | | |

## Consequences

- Why the clamp matters: the clipped cosine schedule ends at abar = 2.4e-9, so DDIM's first step divides the
  network's error by sqrt(abar) ≈ 5e-5 and the predicted x0 explodes. DDPM is not affected because it never forms
  x0 explicitly. A smaller final beta or a "zero terminal SNR" fix would be alternatives; the clamp is one line.
- DDPM's Fréchet distance is at the real-vs-real noise floor, so this judge cannot rank DDPM against real data; the
  shapes task is too easy for that. The metric does separate sampler step counts.
- DDIM with 20 steps is 52x faster than DDPM for a small loss in quality; 10 steps is where the loss becomes visible.
- An exponential moving average of weights with a fixed 0.999 decay produced one shape only after 1,000 steps (it
  still held 37% of the random initialisation); the EMA now warms up.
