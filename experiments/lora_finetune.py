"""Adapt the pretrained GPT to a new corpus: no tuning vs full fine-tuning vs LoRA ranks 2, 8 and 32."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaf.common import ROOT, device, env_info, save, seed_all  # noqa: E402
from gaf.lora import add_lora, trainable  # noqa: E402
from gaf.transformer import GPT, GPTConfig  # noqa: E402
from train_gpt import TEXT_A, TEXT_B, evaluate, split, train  # noqa: E402

STEPS = 400


def load(dev):
    ck = torch.load(ROOT / "checkpoints/gpt_pretrain.pt", map_location="cpu")
    m = GPT(GPTConfig(**ck["cfg"]))
    m.load_state_dict(ck["model"])
    return m.to(dev)


def main():
    dev = device()
    a_tr, a_val = split(TEXT_A)
    b_tr, b_val = split(TEXT_B)
    runs = []
    base = load(dev)
    total = sum(p.numel() for p in base.parameters())
    runs.append({"method": "no fine-tuning", "trainable": 0, "val_new": round(evaluate(base, b_val, dev), 4),
                 "val_original": round(evaluate(base, a_val, dev), 4), "seconds": 0.0})
    for method, r in (("full fine-tuning", None), ("LoRA r=2", 2), ("LoRA r=8", 8), ("LoRA r=32", 32)):
        seed_all(0)
        m = load(dev)
        if r:
            add_lora(m, r=r, alpha=2 * r)
        lr = 3e-4 if r is None else 3e-3
        curve, secs, _, best_step = train(m, b_tr, b_val, dev, STEPS, lr, log_every=50, extra_val=a_val, label=method)
        runs.append({"method": method, "trainable": trainable(m), "trainable_pct": round(100 * trainable(m) / total, 2),
                     "val_new": round(evaluate(m, b_val, dev), 4), "val_original": round(evaluate(m, a_val, dev), 4),
                     "best_step": best_step, "seconds": round(secs, 1), "lr": lr, "curve": curve})
        print(runs[-1]["method"], runs[-1]["trainable"], runs[-1]["val_new"], runs[-1]["val_original"], flush=True)
    save("lora", {"env": env_info(dev), "base_params": total, "steps": STEPS, "new_corpus_chars": len(b_tr),
                  "runs": runs})


if __name__ == "__main__":
    main()
