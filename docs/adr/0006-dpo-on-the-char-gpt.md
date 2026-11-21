# 0006. DPO on the char-level GPT, measured as reward vs KL, with beta >= 1 at this scale

- Status: Accepted (at this scale)
- Date: 2026-11-21

## Context

DPO (Rafailov et al., 2023) trains a policy directly on preference pairs. To measure it exactly, the setup copies
the paper's controlled-sentiment experiment: a known programmatic reward labels pairs sampled from the reference
model. The reward is the share of complete words in a 48-character completion that occur at least twice in the
pretraining corpus (1,271 words), a proxy for "spells real words". 2,048 prompts × 2 samples gave 1,722 pairs
after dropping ties (chosen 0.848, rejected 0.608 on average). Evaluation: 256 held-out prompts × 4 samples at
temperature 1, KL(policy ‖ reference) estimated from the samples, and loss on the original validation text.

## Decision

Report the whole grid (learning rate 1e-5 / 3e-5 / 1e-4 × beta 0.1 / 0.3 / 1.0, 300 steps, batch 64) against two
baselines: filtered SFT (likelihood on the chosen completions only) and best-of-4 sampling from the reference.
No configuration is "selected": the grid is scored on the same prompts, so the best point is optimistic.

## Evidence (`results/dpo.json`)

| Method | Reward (±1 s.e.) | KL, nats | Val loss, original text |
|---|---|---|---|
| Reference | 0.749 ± 0.006 | 0 | 1.516 |
| Best-of-4 sampling | 0.763 | ≤ 0.64 (analytic bound) | 1.516 |
| Filtered SFT, lr 3e-5 (best SFT) | 0.771 ± 0.006 | 11.9 | 1.583 |
| DPO β=1.0, lr 1e-5 (best DPO) | **0.786 ± 0.006** | 11.1 | 1.552 |
| DPO β=0.1, lr 1e-5 | 0.721 ± 0.006 | 17.7 | 1.684 |
| DPO β=0.1, lr 1e-4 | 0.498 ± 0.008 | 73.2 | 2.924 |

## Consequences

- Small beta hurts here. With β=0.1 both the chosen and the rejected completions lose probability
  (−5.2 and −13.8 nats at lr 1e-5): the loss is satisfied by widening the gap, and probability moves to text that is
  neither, which is worse. With β=1.0 the chosen completions gain (+0.7) and only the rejected ones lose (−3.8).
- The best DPO run beats filtered SFT by 0.015 (about 2 standard errors) and the reference by 0.037, with less damage
  to the original-text loss than SFT at a similar KL. Best-of-4 gets a third of the gain at near-zero KL but costs
  4x at inference.
- One run per configuration; MPS kernels are not bit-deterministic, and a re-run moved one grid point by 0.01. The
  reward is a proxy chosen to be measurable, not a human preference.
