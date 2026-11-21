# 0002. Laptop-scale experiments with measured, reproducible results

- Status: Accepted
- Date: 2026-11-21

## Decision

- Every experiment finishes in under a minute on an Apple M4 Pro (GPT on MPS in 48 s; GAN and RL on
  CPU, where small MLPs are faster than on the GPU).
- Fixed seeds. GAN and RL use 3 seeds per setting, because single runs of either are noisy.
- Text comes from this portfolio's own documentation (AGPL-3.0), so no dataset licence or download
  is involved. Pretraining uses repos 1-3. Adaptation uses the ADRs of repos 4-6, which have a different
  vocabulary and style.
- Metrics are chosen so failure is visible: validation loss against the uniform guess, LoRA forgetting
  on the original corpus, GAN modes covered rather than a picture, and episodes until CartPole is solved.
- Failed or overfitting runs are reported in commit messages and the README. They are not deleted.

## Consequences

- The text corpus is small (104k characters), so the transformer overfits. The first run (d_model 192)
  was replaced by a smaller, more regularised one with early stopping; both are documented.
- Notebooks are generated and executed by a script, so the committed outputs are real and the cells
  can be reviewed as Python.
