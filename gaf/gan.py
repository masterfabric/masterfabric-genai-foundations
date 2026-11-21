"""A GAN from scratch (Goodfellow et al., 2014) on a ring of eight 2-D Gaussians.

The toy target makes mode collapse measurable: we count how many of the 8 modes receive generated
samples and what fraction of samples land within 3 standard deviations of any mode.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

STD = 0.05
MODES = torch.tensor([[math.cos(2 * math.pi * i / 8), math.sin(2 * math.pi * i / 8)] for i in range(8)]) * 2.0


def real_batch(n: int, gen: torch.Generator) -> torch.Tensor:
    idx = torch.randint(0, 8, (n,), generator=gen)
    return MODES[idx] + STD * torch.randn(n, 2, generator=gen)


def mlp(i, o, h=128):
    return nn.Sequential(nn.Linear(i, h), nn.LeakyReLU(0.2), nn.Linear(h, h), nn.LeakyReLU(0.2), nn.Linear(h, o))


class GAN(nn.Module):
    def __init__(self, z_dim: int = 8):
        super().__init__()
        self.z_dim = z_dim
        self.G, self.D = mlp(z_dim, 2), mlp(2, 1)

    def sample(self, n, gen=None):
        return self.G(torch.randn(n, self.z_dim, generator=gen))


def quality(samples: torch.Tensor) -> dict:
    d = torch.cdist(samples, MODES)
    near, which = d.min(dim=1)
    good = near < 3 * STD
    counts = torch.bincount(which[good], minlength=8)
    return {"modes_covered": int((counts >= 0.02 * len(samples)).sum()),  # a mode counts if it gets >= 2% of samples
            "high_quality_pct": round(100 * good.float().mean().item(), 1)}


def train(seed: int, steps: int = 4000, loss: str = "non_saturating", bs: int = 256, snapshots=(0, 500, 1500, 4000)):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    m = GAN()
    og = torch.optim.Adam(m.G.parameters(), lr=1e-3, betas=(0.5, 0.999))
    od = torch.optim.Adam(m.D.parameters(), lr=1e-3, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    ones, zeros = torch.ones(bs, 1), torch.zeros(bs, 1)
    curve, snaps = [], {}
    for step in range(steps + 1):
        if step in snapshots:
            with torch.no_grad():
                snaps[step] = m.sample(1000, torch.Generator().manual_seed(99)).tolist()
        if step % 250 == 0:
            with torch.no_grad():
                curve.append({"step": step} | quality(m.sample(2000, torch.Generator().manual_seed(7))))
        if step == steps:
            break
        real, fake = real_batch(bs, gen), m.sample(bs, gen)
        ld = bce(m.D(real), ones) + bce(m.D(fake.detach()), zeros)
        od.zero_grad(); ld.backward(); od.step()
        logits = m.D(m.sample(bs, gen))
        # non-saturating: maximise log D(G(z)); saturating (original minimax): minimise log(1 - D(G(z)))
        lg = bce(logits, ones) if loss == "non_saturating" else -bce(logits, zeros)
        og.zero_grad(); lg.backward(); og.step()
    return {"seed": seed, "loss": loss, "curve": curve, "final": curve[-1], "snapshots": snaps}
