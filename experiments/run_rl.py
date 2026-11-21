"""REINFORCE on CartPole with and without a value baseline, 3 seeds each."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaf.common import save  # noqa: E402
from gaf.rl import train  # noqa: E402

runs = []
t0 = time.perf_counter()
for baseline in (False, True):
    for seed in (0, 1, 2):
        r = train(seed, baseline, updates=120)
        runs.append(r)
        print("baseline" if baseline else "plain", seed, r["final_mean_return"], r["first_update_at_195"], flush=True)
save("rl", {"updates": 120, "envs_per_update": 16, "max_steps": 200, "device": "cpu",
            "seconds": round(time.perf_counter() - t0, 1), "runs": runs})
