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


def _mean(runs, key):
    return sum(r[key] for r in runs) / len(runs)


def ring_compare():
    """Same 8-Gaussian target, same metrics: GAN, WGAN-GP and diffusion side by side."""
    import numpy as np
    g, gl, w, df = J("gan"), J("gan_long"), J("wgan_gp"), J("diffusion")
    modes = np.array(g["modes"])
    ns4 = [r for r in g["runs"] if r["loss"] == "non_saturating"]
    w4, w20 = w["budgets"]["4000"]["runs"], w["budgets"]["20000"]["runs"]

    def fin(runs):
        return (f"{_mean([r['final'] for r in runs], 'modes_covered'):.1f}/8 modes · "
                f"{_mean([r['final'] for r in runs], 'high_quality_pct'):.0f}% on a mode")

    def dif(name):
        rs = [r["samplers"][name] for r in df["ring"]]
        return f"{_mean(rs, 'modes_covered'):.1f}/8 modes · {_mean(rs, 'high_quality_pct'):.0f}% on a mode"
    panels = [
        ("GAN (non-saturating), 4k steps", ns4[0]["snapshots"]["4000"], fin(ns4), HEX["ORANGE"]),
        ("GAN (non-saturating), 20k steps", gl["runs"][0]["snapshots"]["20000"], fin(gl["runs"]), HEX["ORANGE"]),
        ("WGAN-GP, 4k generator steps", w4[0]["snapshots"]["4000"], fin(w4), HEX["PURPLE"]),
        ("WGAN-GP, 20k generator steps", w20[0]["snapshots"]["20000"], fin(w20), HEX["PURPLE"]),
        ("Diffusion, DDPM 1000 steps", df["ring"][0]["samplers"]["DDPM 1000"]["points"], dif("DDPM 1000"), HEX["GREEN"]),
        ("Diffusion, DDIM 10 steps + x0 clamp", df["ring"][0]["samplers"]["DDIM 10 +clip"]["points"], dif("DDIM 10 +clip"), HEX["GREEN"]),
    ]
    fig, axes = plt.subplots(1, 6, figsize=(21, 4.4))
    for a, (title, pts, sub, col) in zip(axes, panels):
        p = np.array(pts)
        a.scatter(p[:, 0], p[:, 1], s=2, color=col, alpha=0.45)
        a.scatter(modes[:, 0], modes[:, 1], s=40, marker="x", color=HEX["RED"])
        a.set_xlim(-3, 3); a.set_ylim(-3, 3); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.set_title(title, loc="left", fontsize=11)
        a.set_xlabel(sub + " (3 seeds)", fontsize=9.5)
    fig.suptitle("One target, three model families: samples of seed 0, mean over 3 seeds below each panel", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(); fig.savefig(IMG / "ring_compare.png"); plt.close(fig)


def diffusion_shapes():
    import numpy as np
    s = J("diffusion")["shapes"]
    grids = [("real (fresh samples)", s["real_grid"], s["real_vs_real"]),
             ("DDPM, 1000 steps", s["samplers"]["DDPM 1000"]["grid"], s["samplers"]["DDPM 1000"]),
             ("DDIM, 50 steps, no clamp", s["samplers"]["DDIM 50"]["grid"], s["samplers"]["DDIM 50"]),
             ("DDIM, 10 steps + x0 clamp", s["samplers"]["DDIM 10 +clip"]["grid"], s["samplers"]["DDIM 10 +clip"])]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2))
    for a, (title, g, m) in zip(axes, grids):
        g = np.array(g)
        tile = np.ones((8 * 18, 8 * 18)) * 40
        for i in range(64):
            r, c = divmod(i, 8)
            tile[r * 18 + 1:r * 18 + 17, c * 18 + 1:c * 18 + 17] = g[i]
        a.imshow(tile, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        a.set_title(title, loc="left")
        shares = " ".join(f"{k[:3]} {v:.0f}%" for k, v in m["class_share_pct"].items())
        a.set_xlabel(f"Fréchet (judge features) {m['frechet_cls']:.2f}\n{shares}", fontsize=10)
    fig.suptitle(f"16x16 synthetic shapes, U-Net {s['params'] / 1e3:.0f}k parameters, {s['train_steps']:,} steps on Apple MPS "
                 f"({s['train_seconds'] / 60:.0f} min) · judge: CNN trained on real shapes only, test accuracy {s['classifier_test_acc']:.3f}",
                 x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(); fig.savefig(IMG / "diffusion_shapes.png"); plt.close(fig)


def diffusion_steps():
    d = J("diffusion")
    names = ["DDIM 10 +clip", "DDIM 20 +clip", "DDIM 50 +clip", "DDIM 100 +clip"]
    steps = [10, 20, 50, 100]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.4))
    hq = [_mean([r["samplers"][n] for r in d["ring"]], "high_quality_pct") for n in names]
    a1.plot(steps, hq, "o-", color=HEX["GREEN"], label="DDIM + x0 clamp")
    a1.axhline(_mean([r["samplers"]["DDPM 1000"] for r in d["ring"]], "high_quality_pct"), color=HEX["ICE"], ls="--",
               label="DDPM, 1000 steps")
    a1.axhline(_mean([r["samplers"]["DDIM 50"] for r in d["ring"]], "high_quality_pct"), color=HEX["RED"], ls=":",
               label="DDIM 50, no clamp")
    a1.set_xscale("log"); a1.set_xticks(steps, [str(x) for x in steps]); a1.set_ylim(0, 100)
    a1.set_xlabel("sampling steps"); a1.set_ylabel("% of samples on a mode (3 seeds)"); a1.legend(fontsize=9, loc="center right")
    a1.set_title("Ring: quality vs sampling steps", loc="left")
    sh = d["shapes"]["samplers"]
    fd = [sh[n]["frechet_cls"] for n in names]
    a2.plot(steps, fd, "o-", color=HEX["GREEN"], label="DDIM + x0 clamp")
    a2.axhline(sh["DDPM 1000"]["frechet_cls"], color=HEX["ICE"], ls="--", label=f"DDPM 1000 ({sh['DDPM 1000']['seconds_2000']:.0f} s / 2,000)")
    a2.axhline(d["shapes"]["real_vs_real"]["frechet_cls"], color=HEX["GREY"], ls=":", label="real vs real (noise floor)")
    for x, y, n in zip(steps, fd, names):
        a2.text(x, y + 0.08, f"{sh[n]['seconds_2000']:.1f} s", ha="center", fontsize=9, color=HEX["GREY"])
    a2.set_xscale("log"); a2.set_xticks(steps, [str(x) for x in steps]); a2.set_ylim(0, max(fd) * 1.3)
    a2.set_xlabel("sampling steps (time for 2,000 images on MPS)"); a2.set_ylabel("Fréchet distance, judge features")
    a2.set_title("Shapes: distance to real images (lower is better)", loc="left"); a2.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(IMG / "diffusion_steps.png"); plt.close(fig)


def dpo():
    d = J("dpo")
    ms = d["methods"]
    ref = ms[0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5))
    marks = {1e-5: "o", 3e-5: "s", 1e-4: "^"}
    cols = {None: HEX["YELLOW"], 0.1: HEX["RED"], 0.3: HEX["ORANGE"], 1.0: HEX["GREEN"]}
    a1.scatter([0], [ref["reward"]], color=HEX["WHITE"], s=70, zorder=3, label="reference")
    b4 = next(m for m in ms if m["method"].startswith("best-of"))
    a1.scatter([b4["kl_nats"]], [b4["reward"]], color=HEX["ICE"], s=70, marker="D", zorder=3, label="best-of-4 (KL bound)")
    for m in ms[2:]:
        a1.scatter(m["kl_nats"], m["reward"], color=cols[m["beta"]], marker=marks[m["lr"]], s=70, zorder=3)
        a1.errorbar(m["kl_nats"], m["reward"], yerr=m["reward_se"], color=cols[m["beta"]], lw=1)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=c, marker="o", ls="", label=("filtered SFT" if b is None else f"DPO β={b}")) for b, c in cols.items()]
    h += [Line2D([], [], color=HEX["GREY"], marker=mk, ls="", label=f"lr {lr:g}") for lr, mk in marks.items()]
    l1 = a1.legend(loc="lower left", fontsize=9)
    a1.add_artist(l1); a1.legend(handles=h, loc="lower right", fontsize=9, ncol=2)
    a1.set_xscale("symlog", linthresh=1); a1.set_xlabel("KL(policy ‖ reference), nats per 48-character completion")
    a1.set_ylabel("reward: share of real words (±1 s.e.)")
    a1.set_title("Reward vs KL on 256 held-out prompts × 4 samples", loc="left")
    runs = [m for m in ms[2:] if m["lr"] == 1e-5]
    x = range(len(runs))
    a2.bar([i - 0.2 for i in x], [m["delta_logp_chosen"] for m in runs], 0.38, color=HEX["GREEN"], label="chosen")
    a2.bar([i + 0.2 for i in x], [m["delta_logp_rejected"] for m in runs], 0.38, color=HEX["RED"], label="rejected")
    a2.axhline(0, color=HEX["GREY"], lw=1)
    a2.set_xticks(list(x), [m["method"].replace(" (chosen only)", "") for m in runs])
    a2.set_ylabel("change in log p of training completions (nats)")
    a2.set_title("Where the probability goes (lr 1e-5)", loc="left"); a2.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(IMG / "dpo.png"); plt.close(fig)


def cfg():
    import numpy as np
    d = J("cfg")
    sc = d["scales"]
    show = ["w=0", "w=1", "w=3", "w=8"]
    names = list(sc["w=1"]["per_class"])
    fig = plt.figure(figsize=(17, 6.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1])
    a0 = fig.add_subplot(gs[0])
    n = 4  # images per class shown
    tile = np.ones((len(show) * 18, 4 * n * 18 + 3 * 8)) * 40
    for r, key in enumerate(show):
        for k in range(4):
            g = np.array(sc[key]["grid"][k])
            for j in range(n):
                c0 = (k * n + j) * 18 + k * 8
                tile[r * 18 + 1:r * 18 + 17, c0 + 1:c0 + 17] = g[j]
    a0.imshow(tile, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    a0.set_yticks([r * 18 + 8.5 for r in range(len(show))],
                  [f"{k.replace('w=', 'w = ')}\n{sc[k]['on_class_pct']:.0f}% on class" for k in show], fontsize=10)
    a0.set_xticks([(k * n + n / 2) * 18 + k * 8 for k in range(4)], [f"asked for: {c}" for c in names], fontsize=10)
    a0.tick_params(length=0); a0.grid(False)
    a0.set_title("Samples per requested class (DDIM 50 + clamp, same noise per row)", loc="left")
    a1 = fig.add_subplot(gs[1])
    ws = [sc[k]["w"] for k in sc]
    cols = {"circle": HEX["ORANGE"], "square": HEX["YELLOW"], "triangle": HEX["GREEN"], "cross": HEX["PURPLE"]}
    a1.plot(ws, [sc[k]["frechet_all"] for k in sc], "o-", color=HEX["WHITE"], lw=2.4, label="all classes")
    for c in names:
        a1.plot(ws[1:], [sc[k]["per_class"][c]["frechet"] for k in list(sc)[1:]], "o--", color=cols[c], lw=1.4,
                label=f"{c}es vs real {c}es" if c == "cross" else f"{c}s vs real {c}s")
    a1.axhline(d["real_vs_real_frechet"], color=HEX["GREY"], ls=":", label=f"real vs real ({d['real_vs_real_frechet']:.2f})")
    a1.set_yscale("log"); a1.set_xlabel("guidance scale w (0 = unconditional, 1 = conditional, >1 = guided)")
    a1.set_ylabel("Fréchet distance to real shapes, judge features (log)")
    a1.set_title("Distance to real images rises with w; on-class share (labels) is full from w = 1", loc="left", fontsize=11)
    a1.legend(fontsize=9, loc="upper left")
    fig.suptitle(f"Classifier-free guidance: one U-Net ({d['params'] / 1e3:.0f}k parameters), {d['train_steps']:,} steps with "
                 f"{d['p_drop']:.0%} label dropout, {d['train_seconds'] / 60:.0f} min on Apple MPS · {d['images_per_class']} images "
                 "per class and scale", x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(); fig.savefig(IMG / "cfg.png"); plt.close(fig)


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    for f in (gpt, lora, gan, rl, ring_compare, diffusion_shapes, diffusion_steps, dpo, cfg):
        try:
            f()
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
    print("charts written")
