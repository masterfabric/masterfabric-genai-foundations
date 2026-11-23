# 0007. Guidance scale chosen by measurement; models are scored in eval mode, checked by a test

- Status: Accepted (at this scale)
- Date: 2026-11-23

## Context

Two follow-ups from the 2026-11-21 round. First, classifier-free guidance (Ho & Salimans, 2022): one
class-conditional U-Net with a null class, trained with 10% label dropout, sampled with
`ε(x, ∅) + w · (ε(x, y) − ε(x, ∅))`. Second, DPO with β chosen on prompts it is not reported on, and a second
round on pairs sampled from the round-1 policy.

While running the second, the reference reward on test prompts came out at 0.562 against 0.734 on selection
prompts drawn from the same text. The cause was `experiments/train_gpt.py: evaluate()`, which ended with
`model.train()`: any model kept in eval mode was switched to training mode, with dropout 0.2, after its validation
loss was measured. In `run_dpo.py` that happened to the reference model after its first measurement, so every later
KL estimate and the best-of-4 samples used a reference with dropout on.

## Decision

- `evaluate()` restores the mode it received; `tests/test_transformer.py::test_evaluate_restores_the_callers_mode`
  fails on the old code and passes on the new.
- Results measured with the bug are kept next to the corrected ones (`results/*_run1_dropout_on.json`) and the
  README says which numbers changed.
- Guidance scale is a measured setting, not a default: on these shapes use w = 1 (plain conditional sampling).

## Evidence (`results/cfg.json`, `results/dpo.json`, `results/dpo_iter.json`)

| Guidance w | Requested class | Fréchet to real (judge features; real vs real 0.80) |
|---|---|---|
| 0 | 23.2% | 1.93 |
| 1 | 99.8% | 0.57 |
| 3 | 100% | 15.5 |
| 8 | 100% | 41.3 |

| DPO, corrected | Reward | KL, nats |
|---|---|---|
| Reference | 0.749 | 0 |
| Best-of-4 sampling | 0.918 | ≤ 0.64 |
| DPO β=1.0, lr 1e-5 | 0.783 | 1.2 (reported earlier: 11.1) |
| Iterative DPO on unseen prompts: reference → round 1 → round 2 | 0.763 → 0.792 → 0.822 | 1.2, then 2.9 to round 1 |

## Consequences

- The DPO section and ADR 0006 now say that best-of-4 wins on this reward; DPO's advantage is no inference cost,
  not a higher reward.
- Guidance above 1 is not worth it on classes the conditional model already gets right; it would need a harder
  image set to show its class-match gain.
- Other experiments call `evaluate()` on models in training mode, which the fix leaves unchanged; they were not
  re-run.
