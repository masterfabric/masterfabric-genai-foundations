"""GAN on the 8-Gaussian ring: saturating vs non-saturating generator loss, 3 seeds each."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaf.common import save  # noqa: E402
from gaf.gan import MODES, train  # noqa: E402

runs = []
t0 = time.perf_counter()
for loss in ("non_saturating", "saturating"):
    for seed in (0, 1, 2):
        r = train(seed, loss=loss)
        if seed != 0:
            r.pop("snapshots")
        runs.append(r)
        print(loss, seed, r["final"], flush=True)
save("gan", {"steps": 4000, "batch": 256, "device": "cpu", "seconds": round(time.perf_counter() - t0, 1),
             "modes": MODES.tolist(), "runs": runs})
