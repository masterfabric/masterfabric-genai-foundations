"""Charts from results/*.json."""
from __future__ import annotations

import json

from mfviz import HEX, IMG, RESULTS, mpl_setup

plt = mpl_setup()


def J(n):
    return json.loads((RESULTS / f"{n}.json").read_text())


def gpt():
    d = J("gpt_pretrain")
    c = d["curve"]
    fig, a = plt.subplots(figsize=(11, 4.6))
    a.plot([r["step"] for r in c], [r["train"] for r in c], "-o", ms=3, color=HEX["ICE"], label="train (batch)")
    a.plot([r["step"] for r in c], [r["val"] for r in c], "-o", ms=3, color=HEX["GREEN"], label="validation")
    a.axhline(d["uniform_loss"], color=HEX["GREY"], ls="--", lw=1)
    a.text(c[-1]["step"], d["uniform_loss"] + 0.05, f"uniform guess {d['uniform_loss']:.2f}", ha="right", fontsize=9, color=HEX["GREY"])
    a.axvline(d["best_step"], color=HEX["YELLOW"], ls=":", lw=1.2)
    a.text(d["best_step"], 3.6, f" best val {d['best_val']:.3f}\n at step {d['best_step']}", fontsize=9, color=HEX["YELLOW"])
    a.set_xlabel("step (batch 64 × 128 characters)"); a.set_ylabel("cross-entropy (nats / char)")
    a.set_title(f"Char-level GPT, {d['params']:,} params, {d['seconds']:.0f} s on Apple {d['env']['device'].upper()} "
                f"({d['tokens_per_s']:,} tok/s)", loc="left")
    a.legend()
    fig.tight_layout(); fig.savefig(IMG / "gpt_loss.png"); plt.close(fig)


def lora():
    d = J("lora")
    runs = d["runs"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 4.8))
    cols = [HEX["GREY"], HEX["ORANGE"], HEX["PURPLE"], HEX["GREEN"], HEX["ICE"]]
    names = [r["method"] for r in runs]
    x = range(len(runs))
    a1.bar([i - 0.2 for i in x], [r["val_new"] for r in runs], 0.38, color=cols, label="new corpus (target)")
    a1.bar([i + 0.2 for i in x], [r["val_original"] for r in runs], 0.38, color=cols, alpha=0.45, hatch="//", label="original corpus (forgetting)")
    for i, r in enumerate(runs):
        a1.text(i - 0.2, r["val_new"] + 0.03, f"{r['val_new']:.2f}", ha="center", fontsize=9)
        a1.text(i + 0.2, r["val_original"] + 0.03, f"{r['val_original']:.2f}", ha="center", fontsize=9)
    a1.set_xticks(list(x), [n.replace(" ", "\n", 1) for n in names]); a1.set_ylabel("validation loss")
    a1.set_title("Adapting to a new corpus: target loss vs forgetting", loc="left"); a1.legend(fontsize=9)
    a1.set_ylim(0, max(max(r["val_new"], r["val_original"]) for r in runs) * 1.2)
    for r, c in zip(runs[1:], cols[1:]):
        a2.plot([p["step"] for p in r["curve"]], [p["val"] for p in r["curve"]], "-o", ms=3, color=c,
                label=f"{r['method']} ({r['trainable']:,} params, {r['trainable_pct']}%)")
    a2.set_xlabel("fine-tuning step"); a2.set_ylabel("validation loss, new corpus")
    a2.set_title("Learning curves", loc="left"); a2.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(IMG / "lora.png"); plt.close(fig)


def gan():
    d = J("gan")
    import numpy as np
    snap = next(r for r in d["runs"] if r["loss"] == "non_saturating" and r["seed"] == 0)["snapshots"]
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.9))
    modes = np.array(d["modes"])
    for a, (step, pts) in zip(axes[:4], sorted(snap.items(), key=lambda kv: int(kv[0]))):
        p = np.array(pts)
        a.scatter(p[:, 0], p[:, 1], s=3, color=HEX["GREEN"], alpha=0.5)
        a.scatter(modes[:, 0], modes[:, 1], s=40, marker="x", color=HEX["RED"])
        a.set_xlim(-3, 3); a.set_ylim(-3, 3); a.set_aspect("equal"); a.set_title(f"step {step}", loc="left")
        a.set_xticks([]); a.set_yticks([])
    a = axes[4]
    for loss, col in (("non_saturating", HEX["GREEN"]), ("saturating", HEX["ORANGE"])):
        for r in (r for r in d["runs"] if r["loss"] == loss):
            a.plot([c["step"] for c in r["curve"]], [c["modes_covered"] for c in r["curve"]], color=col, alpha=0.8,
                   label=loss.replace("_", "-") if r["seed"] == 0 else None)
    a.set_ylim(-0.3, 8.5); a.set_title("modes covered (3 seeds)", loc="left"); a.set_xlabel("step"); a.legend(fontsize=8)
    fig.suptitle("GAN on a ring of 8 Gaussians: generated samples (green) vs true modes (x)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(); fig.savefig(IMG / "gan.png"); plt.close(fig)


def rl():
    d = J("rl")
    fig, a = plt.subplots(figsize=(11, 4.6))
    for bl, col, lab in ((False, HEX["ORANGE"], "REINFORCE"), (True, HEX["GREEN"], "REINFORCE + value baseline")):
        for r in (r for r in d["runs"] if r["baseline"] == bl):
            a.plot([c["episodes"] for c in r["curve"]], [c["mean_return"] for c in r["curve"]], color=col, alpha=0.75,
                   label=lab if r["seed"] == 0 else None)
    a.axhline(195, color=HEX["GREY"], ls="--", lw=1); a.text(50, 198, "solved (195)", fontsize=9, color=HEX["GREY"])
    a.set_xlabel("episodes (16 per update)"); a.set_ylabel("mean episode return (max 200)")
    a.set_title("Policy gradient on CartPole, 3 seeds each", loc="left"); a.legend()
    fig.tight_layout(); fig.savefig(IMG / "rl.png"); plt.close(fig)


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    for f in (gpt, lora, gan, rl):
        try:
            f()
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
    print("charts written")
