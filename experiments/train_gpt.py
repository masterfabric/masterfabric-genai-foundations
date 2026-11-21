"""Pretrain the character-level GPT on documentation from MasterFabric repos 1-3."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaf.common import ROOT, device, env_info, save, seed_all, sync  # noqa: E402
from gaf.transformer import GPT, CharTokenizer, GPTConfig  # noqa: E402

TEXT_A = (ROOT / "data_text/pretrain.txt").read_text()
TEXT_B = (ROOT / "data_text/finetune.txt").read_text()
TOK = CharTokenizer(TEXT_A + TEXT_B)  # one vocabulary for both corpora, so LoRA can reuse the model


def split(text: str, frac: float = 0.9):
    ids = torch.tensor(TOK.encode(text))
    n = int(len(ids) * frac)
    return ids[:n], ids[n:]


def batch(data, bs, block, dev, gen):
    ix = torch.randint(len(data) - block - 1, (bs,), generator=gen)
    x = torch.stack([data[i:i + block] for i in ix])
    y = torch.stack([data[i + 1:i + block + 1] for i in ix])
    return x.to(dev), y.to(dev)


@torch.no_grad()
def evaluate(model, data, dev, iters=20, bs=64):
    model.eval()
    g = torch.Generator().manual_seed(123)
    losses = [model(*batch(data, bs, model.c.block_size, dev, g))[1].item() for _ in range(iters)]
    model.train()
    return sum(losses) / len(losses)


def train(model, train_data, val_data, dev, steps, lr, bs=64, log_every=100, extra_val=None, label=""):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    g = torch.Generator().manual_seed(0)
    curve, t0 = [], time.perf_counter()
    best, best_state = float("inf"), None
    for step in range(1, steps + 1):
        _, loss = model(*batch(train_data, bs, model.c.block_size, dev, g))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % log_every == 0 or step == 1:
            row = {"step": step, "train": round(loss.item(), 4), "val": round(evaluate(model, val_data, dev), 4)}
            if extra_val is not None:
                row["val_other"] = round(evaluate(model, extra_val, dev), 4)
            curve.append(row)
            if row["val"] < best:  # keep the best validation checkpoint (early stopping)
                best, best_step = row["val"], step
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(label, row, flush=True)
    sync(dev)
    secs = time.perf_counter() - t0
    model.load_state_dict(best_state)
    return curve, secs, steps * bs * model.c.block_size / secs, best_step


def main():
    seed_all(0)
    dev = device()
    # A first run (d_model 192, dropout 0.1, 3,000 steps) overfit this 104k-character corpus by step 700
    # (train 0.72, val rising to 1.75); this smaller, more regularised model is the result reported.
    cfg = GPTConfig(vocab_size=len(TOK.chars), d_model=128, dropout=0.2)
    model = GPT(cfg).to(dev)
    a_tr, a_val = split(TEXT_A)
    steps = 1500
    curve, secs, tps, best_step = train(model, a_tr, a_val, dev, steps, lr=2e-3, label="pretrain")
    model.eval()
    seed_all(1)
    prompt = torch.tensor([TOK.encode("## Decision\n\n")], device=dev)
    sample = TOK.decode(model.generate(prompt, 400)[0].tolist())
    params = sum(p.numel() for p in model.parameters())
    Path(ROOT / "checkpoints").mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, ROOT / "checkpoints/gpt_pretrain.pt")
    save("gpt_pretrain", {"env": env_info(dev), "config": cfg.__dict__, "params": params, "train_chars": len(a_tr),
                          "val_chars": len(a_val), "steps": steps, "batch": 64, "seconds": round(secs, 1),
                          "tokens_per_s": round(tps), "uniform_loss": round(float(torch.log(torch.tensor(float(cfg.vocab_size)))), 4),
                          "best_val": min(c["val"] for c in curve), "best_step": best_step, "curve": curve, "sample": sample})
    print(f"params {params:,}  {secs:.0f}s  {tps:,.0f} tok/s")
    print(sample)


if __name__ == "__main__":
    main()
