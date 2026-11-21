"""A GPT-style decoder-only transformer written from the equations.

Attention is computed explicitly (QK^T / sqrt(d), causal mask, softmax, weighted sum of V) rather
than with torch's fused kernel; tests/test_transformer.py checks it against
torch.nn.functional.scaled_dot_product_attention. Only nn.Linear, nn.Embedding and nn.LayerNorm
are taken from PyTorch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 192
    dropout: float = 0.1


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True) -> torch.Tensor:
    """q, k, v: (batch, heads, time, head_dim)."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if causal:
        t = q.size(-2)
        mask = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    return scores.softmax(dim=-1) @ v


class CausalSelfAttention(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.n_head = c.n_head
        self.q, self.k, self.v = (nn.Linear(c.d_model, c.d_model) for _ in range(3))
        self.o = nn.Linear(c.d_model, c.d_model)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        b, t, d = x.shape
        split = lambda y: y.view(b, t, self.n_head, d // self.n_head).transpose(1, 2)  # noqa: E731
        y = attention(split(self.q(x)), split(self.k(x)), split(self.v(x)))
        return self.drop(self.o(y.transpose(1, 2).reshape(b, t, d)))


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.d_model), nn.LayerNorm(c.d_model)
        self.attn = CausalSelfAttention(c)
        self.fc, self.proj = nn.Linear(c.d_model, 4 * c.d_model), nn.Linear(4 * c.d_model, c.d_model)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # pre-norm residual blocks
        return x + self.drop(self.proj(F.gelu(self.fc(self.ln2(x)))))


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab_size, c.d_model)
        self.pos = nn.Embedding(c.block_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        t = idx.size(1)
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln(x))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n: int, temperature: float = 0.8, top_k: int | None = 20):
        for _ in range(n):
            logits, _ = self(idx[:, -self.c.block_size:])
            logits = logits[:, -1] / temperature
            if top_k:
                v, _ = logits.topk(top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            idx = torch.cat([idx, torch.multinomial(logits.softmax(-1), 1)], dim=1)
        return idx


class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids) -> str:
        return "".join(self.chars[i] for i in ids)
