"""REINFORCE (Williams, 1992) with and without a learned value baseline, on CartPole.

CartPole is reimplemented (Barto, Sutton and Anderson, 1983 dynamics, the same constants as the
classic control benchmark) and vectorised in torch, so no Gym dependency is needed.
Episodes are capped at 200 steps.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class CartPole:
    g, mc, mp, l, f, dt = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    x_lim, th_lim, max_steps = 2.4, 12 * 2 * math.pi / 360, 200

    def __init__(self, n: int, gen: torch.Generator):
        self.n, self.gen = n, gen

    def reset(self):
        self.s = (torch.rand(self.n, 4, generator=self.gen) - 0.5) * 0.1
        self.t = 0
        self.alive = torch.ones(self.n, dtype=torch.bool)
        return self.s.clone()

    def step(self, a: torch.Tensor):
        x, xd, th, thd = self.s.unbind(1)
        force = torch.where(a == 1, self.f, -self.f)
        ct, st = torch.cos(th), torch.sin(th)
        tmp = (force + self.mp * self.l * thd ** 2 * st) / (self.mc + self.mp)
        thacc = (self.g * st - ct * tmp) / (self.l * (4 / 3 - self.mp * ct ** 2 / (self.mc + self.mp)))
        xacc = tmp - self.mp * self.l * thacc * ct / (self.mc + self.mp)
        self.s = torch.stack([x + self.dt * xd, xd + self.dt * xacc, th + self.dt * thd, thd + self.dt * thacc], 1)
        self.t += 1
        reward = self.alive.float()  # +1 for every step the pole was still up before this action
        fell = (self.s[:, 0].abs() > self.x_lim) | (self.s[:, 2].abs() > self.th_lim)
        self.alive = self.alive & ~fell
        done = (~self.alive).all() or self.t >= self.max_steps
        return self.s.clone(), reward, done


def net(o):
    return nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, o))


def train(seed: int, baseline: bool, updates: int = 150, n_envs: int = 16, gamma: float = 0.99):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    pi, v = net(2), net(1)
    opt = torch.optim.Adam(list(pi.parameters()) + list(v.parameters()), lr=1e-2)
    env, curve = CartPole(n_envs, gen), []
    for u in range(updates):
        s, logps, rews, masks, states = env.reset(), [], [], [], []
        done = False
        while not done:
            alive = env.alive.clone()
            dist = torch.distributions.Categorical(logits=pi(s))
            a = dist.sample()
            states.append(s); logps.append(dist.log_prob(a)); masks.append(alive.float())
            s, r, done = env.step(a)
            rews.append(r)
        R, returns = torch.zeros(n_envs), []
        for r in reversed(rews):
            R = r + gamma * R
            returns.insert(0, R)
        ret, lp, m = torch.stack(returns), torch.stack(logps), torch.stack(masks)
        ep_return = torch.stack(rews).sum(0)
        if baseline:
            val = v(torch.stack(states)).squeeze(-1)
            adv = (ret - val.detach())
            vloss = (((ret - val) ** 2) * m).sum() / m.sum()
        else:
            adv, vloss = ret, torch.tensor(0.0)
        adv = (adv - (adv * m).sum() / m.sum()) / (adv[m.bool()].std() + 1e-8)
        loss = -(lp * adv * m).sum() / m.sum() + 0.5 * vloss
        opt.zero_grad(); loss.backward(); opt.step()
        curve.append({"update": u + 1, "episodes": (u + 1) * n_envs, "mean_return": round(ep_return.mean().item(), 1)})
    last = [c["mean_return"] for c in curve[-10:]]
    return {"seed": seed, "baseline": baseline, "curve": curve, "final_mean_return": round(sum(last) / len(last), 1),
            "first_update_at_195": next((c["episodes"] for c in curve if c["mean_return"] >= 195), None)}
