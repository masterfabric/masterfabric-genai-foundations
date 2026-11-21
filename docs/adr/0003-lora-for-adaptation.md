# 0003. Use LoRA for domain adaptation when forgetting matters

- Status: Accepted (at this scale; re-measure on a real model)
- Date: 2026-11-21

## Evidence (results/lora.json, 400 steps, best-validation checkpoint)

| Method | Trainable params | Val loss, new corpus | Val loss, original corpus |
|---|---|---|---|
| No fine-tuning | 0 | 2.193 | 1.516 |
| Full fine-tuning | 824,064 (100%) | 1.725 | 1.747 |
| LoRA r=2 | 18,432 (2.2%) | 1.754 | 1.718 |
| LoRA r=8 | 73,728 (9.0%) | 1.758 | 1.642 |
| LoRA r=32 | 294,912 (35.8%) | 1.738 | 1.825 |

## Decision

Prefer LoRA (r around 8) when the base model must stay good at its original domain. It reaches
within 0.03 of full fine-tuning on the new corpus and keeps more of the old one. One adapter per
domain also means one small file per domain, and an exact merge removes any inference cost.

## Consequences

- LoRA was not faster in wall time here (about 20 s vs 15 s): on a 0.8M-parameter model the extra
  matmuls cost more than the smaller optimiser saves. The benefit in memory and speed appears at
  billion-parameter scale, which this lab does not measure.
- Rank 32 overfit fastest and forgot most. Higher rank is not automatically better on small data.
