"""Direct Preference Optimization (Rafailov et al., 2023) from scratch, on the character-level GPT.

DPO fits the policy to preference pairs without a reward model or RL loop:
    L = -log sigmoid( beta * [ (log pi(y_w|x) - log ref(y_w|x)) - (log pi(y_l|x) - log ref(y_l|x)) ] )
where ref is the frozen starting model. As in the paper's controlled-sentiment experiment, the preferences come
from a known programmatic reward, so the result can be measured exactly: here, the share of words in a
completion that are real words of the pretraining corpus ("spelling"). Pairs are two samples from the
reference model for the same prompt; the one with the higher reward is preferred, ties are dropped.
"""
from __future__ import annotations

import re

import torch
import torch.nn.functional as F

WORD = re.compile(r"[A-Za-z]{2,}")


def vocabulary(text: str, min_count: int = 2) -> set[str]:
    from collections import Counter
    c = Counter(w.lower() for w in WORD.findall(text))
    return {w for w, n in c.items() if n >= min_count}


def reward(text: str, vocab: set[str]) -> float:
    """Share of complete words that are in the vocabulary. A word cut off by the end of the sample is ignored."""
    words = WORD.findall(text.rsplit(" ", 1)[0] if " " in text else "")
    return sum(w.lower() in vocab for w in words) / len(words) if words else 0.0


def completion_logprob(model, prompt: torch.Tensor, completion: torch.Tensor) -> torch.Tensor:
    """Sum of log p(completion token | prefix) per row. prompt (B, P), completion (B, C)."""
    x = torch.cat([prompt, completion], 1)
    logits, _ = model(x[:, :-1])
    lp = F.log_softmax(logits[:, prompt.size(1) - 1:], -1)
    return lp.gather(-1, completion.unsqueeze(-1)).squeeze(-1).sum(1)


def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta: float):
    """Returns (loss, preference accuracy of the implicit reward, mean implicit reward margin)."""
    margin = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    return -F.logsigmoid(margin).mean(), (margin > 0).float().mean(), margin.mean()


@torch.no_grad()
def sample(model, prompt: torch.Tensor, n: int, temperature: float = 1.0) -> torch.Tensor:
    """Plain ancestral sampling (no top-k), so KL estimates refer to the model's own distribution."""
    return model.generate(prompt, n, temperature=temperature, top_k=None)[:, prompt.size(1):]
