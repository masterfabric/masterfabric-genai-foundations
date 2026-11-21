"""WGAN-GP on the 8-Gaussian ring, 3 seeds, with the same generator, critic, batch and metrics as run_gan.py.

Two budgets: 4,000 generator steps (the GAN runs' budget) and 20,000, because at 4,000 the curves are still
rising: WGAN-GP's paper settings (lr 1e-4, 5 critic steps) move more slowly than the GAN's lr 1e-3.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaf.common import save  # noqa: E402
from gaf.gan import train_wgan_gp  # noqa: E402

out = {"critic_steps_per_generator_step": 5, "lambda": 10, "lr": 1e-4, "batch": 256, "device": "cpu", "budgets": {}}
for steps in (4000, 20000):
    runs, secs = [], []
    snaps = (0, 500, 1500, 4000) if steps == 4000 else (0, 2000, 8000, 20000)
    for seed in (0, 1, 2):
        t0 = time.perf_counter()
        r = train_wgan_gp(seed, steps=steps, snapshots=snaps)
        secs.append(round(time.perf_counter() - t0, 1))
        if seed != 0:
            r.pop("snapshots")
        runs.append(r)
        print("wgan_gp", steps, seed, r["final"], secs[-1], flush=True)
    out["budgets"][str(steps)] = {"generator_steps": steps, "seconds_per_seed": secs, "runs": runs}
save("wgan_gp", out)
