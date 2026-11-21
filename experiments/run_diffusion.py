"""Diffusion from scratch: (1) the GAN's 8-Gaussian ring, 3 seeds, DDPM vs DDIM step counts; (2) 16x16 synthetic
shapes with a small U-Net on Apple MPS, judged by a classifier trained on real shapes only."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaf.common import device, env_info, save, seed_all, sync  # noqa: E402
from gaf.diffusion import SHAPES, Diffusion, PointDenoiser, ShapeClassifier, UNet, frechet, shapes  # noqa: E402
from gaf.gan import quality, real_batch  # noqa: E402

RING_STEPS, IMG_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 6000, int(sys.argv[2]) if len(sys.argv) > 2 else 8000
# (name, DDIM steps or None for DDPM, clamp predicted x_0 to the data range)
SAMPLERS = [("DDPM 1000", None, False), ("DDIM 50", 50, False), ("DDIM 100 +clip", 100, True), ("DDIM 50 +clip", 50, True),
            ("DDIM 20 +clip", 20, True), ("DDIM 10 +clip", 10, True)]


def ring(seed: int) -> dict:
    seed_all(seed)
    gen = torch.Generator().manual_seed(seed)
    m, dif = PointDenoiser(), Diffusion(1000)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, RING_STEPS)
    curve, t0 = [], time.perf_counter()
    for step in range(1, RING_STEPS + 1):
        loss = dif.loss(m, real_batch(256, gen))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0:
            curve.append({"step": step, "loss": round(loss.item(), 4)})
    train_s = time.perf_counter() - t0
    m.eval()
    out = {"seed": seed, "train_seconds": round(train_s, 1), "curve": curve, "samplers": {}}
    for name, steps, clip in SAMPLERS:
        t = time.perf_counter()
        s = dif.sample(m, (2000, 2), steps, gen=torch.Generator().manual_seed(7), clip=2.5 if clip else None)
        out["samplers"][name] = quality(s) | {"seconds_2000": round(time.perf_counter() - t, 2)}
        if seed == 0:
            out["samplers"][name]["points"] = s[:1000].tolist()
    print("ring", seed, {k: (v["modes_covered"], v["high_quality_pct"]) for k, v in out["samplers"].items()}, flush=True)
    return out


def classifier(dev):
    seed_all(5)
    gen = torch.Generator().manual_seed(5)
    c = ShapeClassifier().to(dev)
    opt = torch.optim.Adam(c.parameters(), lr=1e-3)
    for _ in range(1500):
        x, y = shapes(256, gen)
        loss = torch.nn.functional.cross_entropy(c(x.to(dev)), y.to(dev))
        opt.zero_grad(); loss.backward(); opt.step()
    c.eval()
    x, y = shapes(4000, torch.Generator().manual_seed(999))
    with torch.no_grad():
        acc = (c(x.to(dev)).argmax(1).cpu() == y).float().mean().item()
    return c, acc


@torch.no_grad()
def judge(c, imgs, dev, real_feats):
    imgs = imgs.clamp(-1, 1).to(dev)
    p = c(imgs).softmax(1)
    conf, cls = p.max(1)
    counts = torch.bincount(cls[conf > 0.9].cpu(), minlength=4)
    return {"confident_pct": round(100 * (conf > 0.9).float().mean().item(), 1),
            "class_share_pct": {SHAPES[i]: round(100 * counts[i].item() / len(imgs), 1) for i in range(4)},
            "frechet_cls": round(frechet(c.feats(imgs), real_feats), 3)}


def images(dev):
    c, acc = classifier(dev)
    with torch.no_grad():
        real_a = c.feats(shapes(2000, torch.Generator().manual_seed(1001))[0].to(dev))
        real_b = shapes(2000, torch.Generator().manual_seed(1002))[0]
    seed_all(0)
    gen = torch.Generator().manual_seed(0)
    m, dif = UNet().to(dev), Diffusion(1000, dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=IMG_STEPS, pct_start=0.05)
    ema = {k: v.detach().clone() for k, v in m.state_dict().items()}
    curve, t0 = [], time.perf_counter()
    for step in range(1, IMG_STEPS + 1):
        x, _ = shapes(128, gen)  # fresh images every step: the model cannot memorise a fixed training set
        loss = dif.loss(m, x.to(dev))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        d = min(0.999, (1 + step) / (10 + step))  # EMA with warm-up; a fixed 0.999 kept 37% of the random init at step 1,000
        with torch.no_grad():
            for k, v in m.state_dict().items():
                ema[k].mul_(d).add_(v.detach(), alpha=1 - d) if v.dtype.is_floating_point else ema[k].copy_(v)
        if step % 500 == 0:
            curve.append({"step": step, "loss": round(loss.item(), 4), "seconds": round(time.perf_counter() - t0, 1)})
            print("unet", curve[-1], flush=True)
    sync(dev)
    train_s = time.perf_counter() - t0
    m.load_state_dict(ema)
    m.eval()
    out = {"params": sum(p.numel() for p in m.parameters()), "train_steps": IMG_STEPS, "batch": 128,
           "train_seconds": round(train_s, 1), "classifier_test_acc": round(acc, 4), "curve": curve,
           "real_vs_real": judge(c, real_b, dev, real_a), "samplers": {}}
    for name, steps, clip in SAMPLERS:
        t = time.perf_counter()
        imgs = torch.cat([dif.sample(m, (500, 1, 16, 16), steps, dev, torch.Generator().manual_seed(40 + i), 1.0 if clip else None)
                          for i in range(4)])
        sync(dev)
        out["samplers"][name] = judge(c, imgs, dev, real_a) | {"seconds_2000": round(time.perf_counter() - t, 1)}
        if name in ("DDPM 1000", "DDIM 50", "DDIM 50 +clip", "DDIM 10 +clip"):
            out["samplers"][name]["grid"] = imgs[:64, 0].clamp(-1, 1).cpu().mul(127.5).add(127.5).round().int().tolist()
        print("unet", name, {k: v for k, v in out["samplers"][name].items() if k != "grid"}, flush=True)
    out["real_grid"] = real_b[:64, 0].mul(127.5).add(127.5).round().int().tolist()
    return out


def main():
    dev = device()
    t0 = time.perf_counter()
    rings = [ring(s) for s in (0, 1, 2)]
    img = images(dev)
    save("diffusion", {"env": env_info(dev), "T": 1000, "schedule": "cosine", "ring_steps": RING_STEPS,
                       "ring": rings, "shapes": img, "seconds": round(time.perf_counter() - t0, 1)})


if __name__ == "__main__":
    main()
