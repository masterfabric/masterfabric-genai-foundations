"""A denoising diffusion model from scratch (Ho et al., 2020), with DDIM sampling (Song et al., 2021).

Forward process: x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps. The network predicts eps from (x_t, t) and is
trained with plain MSE. Sampling runs the reverse chain: all T ancestral steps (DDPM), or a deterministic
subsequence of S steps (DDIM, eta = 0). Two denoisers: an MLP for 2-D points (the GAN's ring of 8 Gaussians, so
the numbers are directly comparable) and a small U-Net for 16x16 images of synthetic shapes.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """abar_t from Nichol and Dhariwal (2021); betas clipped at 0.999."""
    t = torch.linspace(0, T, T + 1, dtype=torch.float64)
    f = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    abar = f / f[0]
    betas = (1 - abar[1:] / abar[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, 0).float()


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=1)


class Diffusion:
    def __init__(self, T: int = 1000, device="cpu"):
        self.T = T
        self.abar = cosine_schedule(T).to(device)

    def q_sample(self, x0, t, eps):
        a = self.abar[t].view(-1, *([1] * (x0.dim() - 1)))
        return a.sqrt() * x0 + (1 - a).sqrt() * eps

    def loss(self, model, x0, gen=None):
        t = torch.randint(0, self.T, (x0.size(0),), device=x0.device)
        eps = torch.randn(x0.shape, device=x0.device)
        return F.mse_loss(model(self.q_sample(x0, t, eps), t), eps)

    @torch.no_grad()
    def sample(self, model, shape, steps: int | None = None, device="cpu", gen=None, clip: float | None = None):
        """steps=None: DDPM ancestral sampling over all T steps. steps=S: DDIM (eta 0) on S evenly spaced steps.

        clip: clamp DDIM's predicted x_0 to [-clip, clip] and recompute eps from it. With the cosine schedule the
        last abar is 2.4e-9, so the first DDIM step divides the network's error by sqrt(abar) ~ 5e-5; without
        the clamp the predicted x_0 explodes (measured in results/diffusion.json)."""
        x = torch.randn(shape, generator=gen).to(device)
        if steps is None:
            for t in range(self.T - 1, -1, -1):
                tt = torch.full((shape[0],), t, device=device, dtype=torch.long)
                a = self.abar[t]
                a_prev = self.abar[t - 1] if t > 0 else torch.tensor(1.0, device=device)
                beta = 1 - a / a_prev
                eps = model(x, tt)
                mean = (x - beta / (1 - a).sqrt() * eps) / (1 - beta).sqrt()
                if t > 0:
                    var = beta * (1 - a_prev) / (1 - a)  # posterior variance
                    x = mean + var.sqrt() * torch.randn(shape, generator=gen).to(device)
                else:
                    x = mean
            return x
        ts = torch.linspace(self.T - 1, 0, steps).round().long().tolist()
        for i, t in enumerate(ts):
            tt = torch.full((shape[0],), t, device=device, dtype=torch.long)
            a = self.abar[t]
            a_prev = self.abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
            eps = model(x, tt)
            x0 = (x - (1 - a).sqrt() * eps) / a.sqrt()
            if clip is not None:
                x0 = x0.clamp(-clip, clip)
                eps = (x - a.sqrt() * x0) / (1 - a).sqrt()
            x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        return x


class PointDenoiser(nn.Module):
    def __init__(self, h: int = 256, emb: int = 64):
        super().__init__()
        self.emb = emb
        self.net = nn.Sequential(nn.Linear(2 + emb, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                                 nn.Linear(h, 2))

    def forward(self, x, t):
        return self.net(torch.cat([x, time_embedding(t, self.emb)], 1))


class ResBlock(nn.Module):
    def __init__(self, cin, cout, temb):
        super().__init__()
        self.n1, self.n2 = nn.GroupNorm(8, cin), nn.GroupNorm(8, cout)
        self.c1, self.c2 = nn.Conv2d(cin, cout, 3, padding=1), nn.Conv2d(cout, cout, 3, padding=1)
        self.t = nn.Linear(temb, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, te):
        h = self.c1(F.silu(self.n1(x))) + self.t(te)[:, :, None, None]
        return self.skip(x) + self.c2(F.silu(self.n2(h)))


class UNet(nn.Module):
    """16x16 -> 8x8 -> 4x4 and back, with skip connections; 582k parameters at ch=32."""

    def __init__(self, ch: int = 32, temb: int = 128):
        super().__init__()
        self.temb = temb
        self.tmlp = nn.Sequential(nn.Linear(temb, temb), nn.SiLU(), nn.Linear(temb, temb))
        self.inp = nn.Conv2d(1, ch, 3, padding=1)
        self.d1, self.d2 = ResBlock(ch, ch, temb), ResBlock(ch, 2 * ch, temb)
        self.mid = ResBlock(2 * ch, 4 * ch, temb)
        self.u2, self.u1 = ResBlock(4 * ch + 2 * ch, 2 * ch, temb), ResBlock(2 * ch + ch, ch, temb)
        self.out = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, 1, 3, padding=1))

    def forward(self, x, t):
        te = self.tmlp(time_embedding(t, self.temb))
        h0 = self.d1(self.inp(x), te)                      # 16x16
        h1 = self.d2(F.avg_pool2d(h0, 2), te)              # 8x8
        m = self.mid(F.avg_pool2d(h1, 2), te)              # 4x4
        u = self.u2(torch.cat([F.interpolate(m, scale_factor=2), h1], 1), te)
        u = self.u1(torch.cat([F.interpolate(u, scale_factor=2), h0], 1), te)
        return self.out(u)


SHAPES = ("circle", "square", "triangle", "cross")


def shapes(n: int, gen: torch.Generator, size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """n filled shapes, random class, centre and size, values in [-1, 1]; 4x supersampled for soft edges."""
    s = size * 4
    yy, xx = torch.meshgrid(torch.arange(s) + 0.5, torch.arange(s) + 0.5, indexing="ij")
    labels = torch.randint(0, 4, (n,), generator=gen)
    r = (torch.rand(n, generator=gen) * 0.15 + 0.2) * s          # half-size 20-35% of the image
    cx = s / 2 + (torch.rand(n, generator=gen) - 0.5) * (s - 2 * r) * 0.9
    cy = s / 2 + (torch.rand(n, generator=gen) - 0.5) * (s - 2 * r) * 0.9
    X, Y = xx[None] - cx[:, None, None], yy[None] - cy[:, None, None]
    R = r[:, None, None]
    masks = torch.stack([
        X ** 2 + Y ** 2 <= R ** 2,
        (X.abs() <= R * 0.85) & (Y.abs() <= R * 0.85),
        (Y <= R * 0.8) & (Y >= -R) & (X.abs() <= (Y + R) * 0.55),
        ((X.abs() <= R * 0.3) & (Y.abs() <= R)) | ((Y.abs() <= R * 0.3) & (X.abs() <= R)),
    ], 1)
    img = masks[torch.arange(n), labels].float()
    img = F.avg_pool2d(img[:, None], 4)
    return img * 2 - 1, labels


class ShapeClassifier(nn.Module):
    """Judge for generated images: trained on real shapes only. feats() is used for a Frechet distance."""

    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1),
                               nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(64 * 16, 128), nn.ReLU())
        self.head = nn.Linear(128, 4)

    def feats(self, x):
        return self.f(x)

    def forward(self, x):
        return self.head(self.f(x))


def frechet(a: torch.Tensor, b: torch.Tensor) -> float:
    """Frechet distance between Gaussians fitted to two feature sets (the FID formula, own features)."""
    import numpy as np
    from numpy.linalg import eigvals
    a, b = a.cpu().double().numpy(), b.cpu().double().numpy()
    m1, m2 = a.mean(0), b.mean(0)
    s1, s2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    tr_sqrt = np.sqrt(np.clip(eigvals(s1 @ s2).real, 0, None)).sum()
    return float(((m1 - m2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * tr_sqrt)
