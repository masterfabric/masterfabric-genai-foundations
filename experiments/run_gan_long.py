"""Control for the WGAN-GP comparison: the non-saturating GAN of run_gan.py trained for 20,000 generator steps too,
so a difference at 20,000 steps is not just more training."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaf.common import save  # noqa: E402
from gaf.gan import train  # noqa: E402

runs, secs = [], []
for seed in (0, 1, 2):
    t0 = time.perf_counter()
    r = train(seed, steps=20000, snapshots=(20000,))
    secs.append(round(time.perf_counter() - t0, 1))
    if seed != 0:
        r.pop("snapshots")
    runs.append(r)
    print("non_saturating 20000", seed, r["final"], secs[-1], flush=True)
save("gan_long", {"loss": "non_saturating", "generator_steps": 20000, "batch": 256, "device": "cpu",
                  "seconds_per_seed": secs, "runs": runs})
