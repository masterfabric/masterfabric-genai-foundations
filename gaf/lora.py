"""LoRA (Hu et al., 2021) from scratch: W x + (alpha / r) * B A x, with W frozen, A random, B zero."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))  # zero: the adapter starts as a no-op
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        """Fold the adapter into a plain Linear for inference (no extra latency)."""
        out = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None)
        out.weight.copy_(self.base.weight + self.scale * self.B @ self.A)
        if self.base.bias is not None:
            out.bias.copy_(self.base.bias)
        return out.to(self.base.weight.device)


def add_lora(model: nn.Module, r: int = 8, alpha: float = 16.0, targets=("q", "k", "v", "o", "fc", "proj")) -> nn.Module:
    for p in model.parameters():
        p.requires_grad_(False)
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                setattr(module, name, LoRALinear(child, r, alpha))
    return model


def trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
