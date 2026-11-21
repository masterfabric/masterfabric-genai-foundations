"""DPO on the pretrained char-level GPT, against two baselines: best-of-4 sampling from the reference model, and
supervised fine-tuning on the preferred completions only ("filtered SFT"). Every method is measured by the same
reward on held-out prompts, by its KL divergence from the reference, and by loss on the original validation text
(how much the language model is damaged)."""
from __future__ import annotations

import copy
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gaf.common import ROOT, device, env_info, save, seed_all, sync  # noqa: E402
from gaf.dpo import completion_logprob, dpo_loss, reward, sample, vocabulary  # noqa: E402
from gaf.transformer import GPT, GPTConfig  # noqa: E402
from train_gpt import TEXT_A, TOK, evaluate, split  # noqa: E402

P, C = 32, 48              # prompt and completion length in characters
N_TRAIN, N_EVAL, K_EVAL = 2048, 256, 4
STEPS, BS = 300, 64
LRS = (1e-5, 3e-5, 1e-4)
BETAS = (0.1, 0.3, 1.0)
# A first sweep at lr 1e-4 with beta 0.03/0.1/0.3 wrecked the model (reward 0.35-0.65 vs 0.75 before, KL 37-111
# nats, validation loss 2.1-4.0 vs 1.52). The grid below adds smaller learning rates and a larger beta.


def load(dev):
    ck = torch.load(ROOT / "checkpoints/gpt_pretrain.pt", map_location="cpu")
    m = GPT(GPTConfig(**ck["cfg"]))
    m.load_state_dict(ck["model"])
    return m.to(dev).eval()  # eval mode throughout: no dropout, so policy and reference log-probs are comparable


def prompts(data, n, gen):
    ix = torch.randint(len(data) - P - 1, (n,), generator=gen)
    return torch.stack([data[i:i + P] for i in ix])


def texts(ids):
    return [TOK.decode(r.tolist()) for r in ids]


@torch.no_grad()
def measure(model, ref, eval_prompts, vocab, a_val, dev, gen_seed=11):
    torch.manual_seed(gen_seed)
    rs, kls = [], []
    for k in range(K_EVAL):
        for i in range(0, len(eval_prompts), 128):
            p = eval_prompts[i:i + 128].to(dev)
            c = sample(model, p, C)
            rs += [reward(t, vocab) for t in texts(c)]
            kls.append((completion_logprob(model, p, c) - completion_logprob(ref, p, c)).cpu())
    model.train(False)
    mean = sum(rs) / len(rs)
    se = (sum((r - mean) ** 2 for r in rs) / (len(rs) - 1) / len(rs)) ** 0.5
    return {"reward": round(mean, 4), "reward_se": round(se, 4), "kl_nats": round(torch.cat(kls).mean().item(), 3),
            "val_loss_original": round(evaluate(model, a_val, dev), 4)}


def main():
    seed_all(0)
    dev = device()
    a_tr, a_val = split(TEXT_A)
    vocab = vocabulary(TOK.decode(a_tr.tolist()))
    ref = load(dev)
    for p in ref.parameters():
        p.requires_grad_(False)
    gen = torch.Generator().manual_seed(0)
    train_p, eval_p = prompts(a_tr, N_TRAIN, gen), prompts(a_val, N_EVAL, gen)
    # 1. preference pairs from the reference model
    t0 = time.perf_counter()
    torch.manual_seed(1)
    ya, yb = [], []
    for i in range(0, N_TRAIN, 256):
        p = train_p[i:i + 256].to(dev)
        ya.append(sample(ref, p, C).cpu()); yb.append(sample(ref, p, C).cpu())
    ya, yb = torch.cat(ya), torch.cat(yb)
    ra, rb = torch.tensor([reward(t, vocab) for t in texts(ya)]), torch.tensor([reward(t, vocab) for t in texts(yb)])
    keep = ra != rb
    win = torch.where((ra > rb)[:, None], ya, yb)[keep]
    lose = torch.where((ra > rb)[:, None], yb, ya)[keep]
    pp = train_p[keep]
    pairs = {"sampled": N_TRAIN, "kept_after_ties": int(keep.sum()), "seconds": round(time.perf_counter() - t0, 1),
             "mean_reward_chosen": round(torch.maximum(ra, rb)[keep].mean().item(), 4),
             "mean_reward_rejected": round(torch.minimum(ra, rb)[keep].mean().item(), 4)}
    print("pairs", pairs, flush=True)
    with torch.no_grad():
        ref_w = torch.cat([completion_logprob(ref, pp[i:i + 256].to(dev), win[i:i + 256].to(dev)).cpu() for i in range(0, len(pp), 256)])
        ref_l = torch.cat([completion_logprob(ref, pp[i:i + 256].to(dev), lose[i:i + 256].to(dev)).cpu() for i in range(0, len(pp), 256)])
    out = {"env": env_info(dev), "prompt_chars": P, "completion_chars": C, "vocabulary_words": len(vocab),
           "pairs": pairs, "steps": STEPS, "batch": BS, "lrs": LRS, "eval_prompts": N_EVAL, "samples_per_prompt": K_EVAL,
           "methods": []}
    base = measure(ref, ref, eval_p, vocab, a_val, dev)
    out["methods"].append({"method": "reference (pretrained)", "beta": None} | base)
    print(out["methods"][-1], flush=True)
    # best-of-4 from the reference: KL upper bound log n - (n-1)/n (Beirami et al., 2024)
    torch.manual_seed(11)
    best = []
    for i in range(0, N_EVAL, 128):
        p = eval_p[i:i + 128].to(dev)
        cands = [[reward(t, vocab) for t in texts(sample(ref, p, C))] for _ in range(4)]
        best += [max(col) for col in zip(*cands)]
    out["methods"].append({"method": "best-of-4 sampling", "beta": None, "reward": round(sum(best) / len(best), 4),
                           "kl_nats": round(math.log(4) - 3 / 4, 3), "kl_note": "analytic bound, not measured",
                           "val_loss_original": base["val_loss_original"], "inference_cost": "4x"})
    print(out["methods"][-1], flush=True)

    def run(name, beta, lr):
        seed_all(0)
        pol = load(dev)
        opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=0.0)
        g = torch.Generator().manual_seed(2)
        curve, t0 = [], time.perf_counter()
        for step in range(1, STEPS + 1):
            ix = torch.randint(len(pp), (BS,), generator=g)
            p, w, lo = pp[ix].to(dev), win[ix].to(dev), lose[ix].to(dev)
            if beta is None:   # filtered SFT: maximise log-likelihood of the preferred completion only
                loss = -completion_logprob(pol, p, w).mean() / C
                acc = margin = torch.tensor(float("nan"))
            else:
                loss, acc, margin = dpo_loss(completion_logprob(pol, p, w), completion_logprob(pol, p, lo),
                                             ref_w[ix].to(dev), ref_l[ix].to(dev), beta)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
            if step % 50 == 0 or step == 1:
                curve.append({"step": step, "loss": round(loss.item(), 4), "pref_acc": round(acc.item(), 3),
                              "margin": round(margin.item(), 3)})
        sync(dev)
        secs = time.perf_counter() - t0
        with torch.no_grad():  # likelihood displacement: how log p of the training pairs moved (first 512 pairs)
            p, w, lo = pp[:512].to(dev), win[:512].to(dev), lose[:512].to(dev)
            disp = {"delta_logp_chosen": round((completion_logprob(pol, p, w).cpu() - ref_w[:512]).mean().item(), 2),
                    "delta_logp_rejected": round((completion_logprob(pol, p, lo).cpu() - ref_l[:512]).mean().item(), 2)}
        r = {"method": name, "beta": beta, "lr": lr} | measure(pol, ref, eval_p, vocab, a_val, dev) | disp | {"train_seconds": round(secs, 1),
                                                                                                  "curve": curve}
        torch.manual_seed(5)
        demo = torch.tensor([TOK.encode("The pipeline writes the results ")], device=dev)[:, :P]
        r["example"] = TOK.decode(sample(pol, demo, C)[0].tolist())
        print({k: v for k, v in r.items() if k != "curve"}, flush=True)
        return r

    for lr in LRS:
        out["methods"].append(run("filtered SFT (chosen only)", None, lr))
        for b in BETAS:
            out["methods"].append(run(f"DPO beta={b}", b, lr))
    torch.manual_seed(5)
    demo = torch.tensor([TOK.encode("The pipeline writes the results ")], device=dev)[:, :P]
    out["methods"][0]["example"] = TOK.decode(sample(ref, demo, C)[0].tolist())
    save("dpo", out)


if __name__ == "__main__":
    main()
