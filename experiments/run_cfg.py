"""Classifier-free guidance on the 16x16 shapes: one class-conditional U-Net trained with 10% label dropout, sampled
at guidance scales w = 0 (unconditional), 1 (conditional), 2, 3, 5 and 8. Measured with the same judge as
run_diffusion.py (a classifier trained on real shapes only): how often the image is the requested class, how often
the judge is confident, the Frechet distance of judge features to real shapes (overall and per class), and pixel
diversity within a class. Usage: python experiments/run_cfg.py [train_steps] [images_per_class]"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gaf.common import device, env_info, save, seed_all, sync  # noqa: E402
from gaf.diffusion import SHAPES, CondUNet, Diffusion, frechet, guided, shapes  # noqa: E402
from run_diffusion import classifier  # noqa: E402

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
PER_CLASS = int(sys.argv[2]) if len(sys.argv) > 2 else 500
P_DROP = 0.1
SCALES = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0)
DDIM_STEPS = 50  # DDIM with the x0 clamp, the sampler ADR 0005 chose


def train(dev):
    seed_all(0)
    gen = torch.Generator().manual_seed(0)
    m, dif = CondUNet().to(dev), Diffusion(1000, dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=STEPS, pct_start=0.05)
    ema = {k: v.detach().clone() for k, v in m.state_dict().items()}
    curve, t0 = [], time.perf_counter()
    for step in range(1, STEPS + 1):
        x, y = shapes(128, gen)
        x, y = x.to(dev), y.to(dev)
        drop = torch.rand(y.shape, generator=gen).to(dev) < P_DROP
        y = torch.where(drop, torch.full_like(y, m.n_classes), y)
        loss = dif.loss(lambda xt, t: m(xt, t, y), x)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        d = min(0.999, (1 + step) / (10 + step))
        with torch.no_grad():
            for k, v in m.state_dict().items():
                ema[k].mul_(d).add_(v.detach(), alpha=1 - d) if v.dtype.is_floating_point else ema[k].copy_(v)
        if step % 500 == 0:
            curve.append({"step": step, "loss": round(loss.item(), 4), "seconds": round(time.perf_counter() - t0, 1)})
            print("cfg", curve[-1], flush=True)
    sync(dev)
    m.load_state_dict(ema)
    return m.eval(), dif, round(time.perf_counter() - t0, 1), curve


@torch.no_grad()
def evaluate(m, dif, c, dev, real_feats, real_by_class):
    out = {}
    for w in SCALES:
        t0 = time.perf_counter()
        imgs, ys = [], []
        for k in range(4):
            y = torch.full((PER_CLASS,), k, device=dev, dtype=torch.long)
            imgs.append(dif.sample(guided(m, y, w), (PER_CLASS, 1, 16, 16), DDIM_STEPS, dev,
                                   torch.Generator().manual_seed(100 + k), 1.0).clamp(-1, 1))
            ys.append(torch.full((PER_CLASS,), k))
        sync(dev)
        secs = time.perf_counter() - t0
        x, y = torch.cat(imgs), torch.cat(ys)
        p = c(x.to(dev)).softmax(1).cpu()
        conf, cls = p.max(1)
        feats = c.feats(x.to(dev))
        per = {}
        for k in range(4):
            sel = y == k
            per[SHAPES[k]] = {"on_class_pct": round(100 * (cls[sel] == k).float().mean().item(), 1),
                              "frechet": round(frechet(feats[sel.to(dev)], real_by_class[k]), 3),
                              "pixel_std": round(x[sel.to(x.device)].std(0).mean().item(), 4)}
        out[f"w={w:g}"] = {"w": w, "on_class_pct": round(100 * (cls == y).float().mean().item(), 1),
                           "confident_pct": round(100 * (conf > 0.9).float().mean().item(), 1),
                           "frechet_all": round(frechet(feats, real_feats), 3),
                           "pixel_std_within_class": round(sum(v["pixel_std"] for v in per.values()) / 4, 4),
                           "seconds": round(secs, 1), "per_class": per,
                           "grid": [x[k * PER_CLASS:k * PER_CLASS + 8, 0].mul(127.5).add(127.5).round().int().tolist()
                                    for k in range(4)]}
        print("cfg", f"w={w:g}", {k: v for k, v in out[f"w={w:g}"].items() if k not in ("grid", "per_class")}, flush=True)
    return out


def main():
    dev = device()
    c, acc = classifier(dev)
    with torch.no_grad():
        rx, ry = shapes(4 * PER_CLASS, torch.Generator().manual_seed(1001))
        f = c.feats(rx.to(dev))
        real_feats = f
        real_by_class = [f[(ry == k).to(dev)] for k in range(4)]
        rx2, ry2 = shapes(4 * PER_CLASS, torch.Generator().manual_seed(1002))
        f2 = c.feats(rx2.to(dev))
        real_vs_real = round(frechet(f2, real_feats), 3)
        real_std = round(sum(rx2[ry2 == k].std(0).mean().item() for k in range(4)) / 4, 4)
    m, dif, train_s, curve = train(dev)
    res = evaluate(m, dif, c, dev, real_feats, real_by_class)
    save("cfg", {"env": env_info(dev), "train_steps": STEPS, "batch": 128, "p_drop": P_DROP, "ddim_steps": DDIM_STEPS,
                 "images_per_class": PER_CLASS, "params": sum(p.numel() for p in m.parameters()),
                 "train_seconds": train_s, "curve": curve, "judge_test_acc": round(acc, 4),
                 "real_vs_real_frechet": real_vs_real, "real_pixel_std_within_class": real_std, "scales": res})


if __name__ == "__main__":
    main()
