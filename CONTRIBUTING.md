# Contributing

This is a lab repository. Changes are welcome when they keep it runnable on a laptop and keep every
number in the README reproducible.

## Ground rules

- **From scratch means from scratch.** New building blocks are written from their equations using
  PyTorch tensors and autograd, not imported from a model library. Add a test that checks them
  against a reference where one exists, like the attention vs SDPA test.
- **Measured numbers only.** If a change affects results, re-run the experiment (`make <target>`),
  rebuild the notebooks and charts (`make notebooks charts`), and commit the new `results/*.json`.
  Never edit a results file or a notebook output by hand.
- **Keep failures on record.** Say in the commit body when a run overfit, diverged or collapsed,
  and keep the metric that shows it.
- **Small and fast.** Every experiment should finish in a few minutes on a laptop.
- No checkpoints, datasets or model weights in git (`checkpoints/` is ignored).
- One logical change per commit, Conventional Commit subject, and a body that explains why.

## License of contributions

By contributing you agree that your contribution is licensed under the AGPL-3.0, as described in
[LICENSE](LICENSE) and [NOTICE](NOTICE).
