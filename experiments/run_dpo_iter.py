"""DPO, second look: (1) choose beta on separate selection prompts and report on unseen test prompts (the first run
reported its grid on the prompts it was chosen on); (2) iterative DPO: a second round on fresh pairs sampled from
the round-1 policy, with the round-1 policy as the new reference. Same reward, model and settings as run_dpo.py."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gaf.common import device, env_info, save, seed_all, sync  # noqa: E402
from gaf.dpo import completion_logprob, dpo_loss, reward, sample, vocabulary  # noqa: E402
from run_dpo import BS, C, N_TRAIN, STEPS, TOK, load, measure, prompts, texts  # noqa: E402
from train_gpt import TEXT_A, split  # noqa: E402

LR = 1e-5
BETAS = (0.3, 1.0, 3.0)
N_SELECT, N_TEST = 128, 256


def make_pairs(policy, train_p, vocab, dev, seed):
    torch.manual_seed(seed)
    ya, yb = [], []
    for i in range(0, len(train_p), 256):
        p = train_p[i:i + 256].to(dev)
        ya.append(sample(policy, p, C).cpu()); yb.append(sample(policy, p, C).cpu())
    ya, yb = torch.cat(ya), torch.cat(yb)
    ra = torch.tensor([reward(t, vocab) for t in texts(ya)])
    rb = torch.tensor([reward(t, vocab) for t in texts(yb)])
    keep = ra != rb
    win = torch.where((ra > rb)[:, None], ya, yb)[keep]
    lose = torch.where((ra > rb)[:, None], yb, ya)[keep]
    info = {"sampled": len(train_p), "kept_after_ties": int(keep.sum()),
            "mean_reward_chosen": round(torch.maximum(ra, rb)[keep].mean().item(), 4),
            "mean_reward_rejected": round(torch.minimum(ra, rb)[keep].mean().item(), 4)}
    return train_p[keep], win, lose, info


def train(start_state, ref, pp, win, lose, beta, dev):
    seed_all(0)
    pol = load(dev)
    pol.load_state_dict(start_state)
    with torch.no_grad():
        ref_w = torch.cat([completion_logprob(ref, pp[i:i + 256].to(dev), win[i:i + 256].to(dev)).cpu() for i in range(0, len(pp), 256)])
        ref_l = torch.cat([completion_logprob(ref, pp[i:i + 256].to(dev), lose[i:i + 256].to(dev)).cpu() for i in range(0, len(pp), 256)])
    opt = torch.optim.AdamW(pol.parameters(), lr=LR, weight_decay=0.0)
    g = torch.Generator().manual_seed(2)
    t0 = time.perf_counter()
    for _ in range(STEPS):
        ix = torch.randint(len(pp), (BS,), generator=g)
        p, w, lo = pp[ix].to(dev), win[ix].to(dev), lose[ix].to(dev)
        loss, _, _ = dpo_loss(completion_logprob(pol, p, w), completion_logprob(pol, p, lo), ref_w[ix].to(dev),
                              ref_l[ix].to(dev), beta)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
    sync(dev)
    return pol.eval(), round(time.perf_counter() - t0, 1)


def main():
    seed_all(0)
    dev = device()
    a_tr, a_val = split(TEXT_A)
    vocab = vocabulary(TOK.decode(a_tr.tolist()))
    base = load(dev)
    for p in base.parameters():
        p.requires_grad_(False)
    gen = torch.Generator().manual_seed(0)
    train_p1 = prompts(a_tr, N_TRAIN, gen)
    gen2 = torch.Generator().manual_seed(123)
    select_p, test_p = prompts(a_val, N_SELECT, gen2), prompts(a_val, N_TEST, gen2)
    train_p2 = prompts(a_tr, N_TRAIN, torch.Generator().manual_seed(77))
    out = {"env": env_info(dev), "lr": LR, "betas": BETAS, "steps": STEPS, "select_prompts": N_SELECT,
           "test_prompts": N_TEST, "rows": []}

    def row(name, model, extra=None):
        r = {"method": name} | {f"select_{k}": v for k, v in measure(model, base, select_p, vocab, a_val, dev).items()
                                if k == "reward"} | measure(model, base, test_p, vocab, a_val, dev) | (extra or {})
        out["rows"].append(r)
        print(r, flush=True)

    row("reference (pretrained)", base)
    pp, win, lose, info1 = make_pairs(base, train_p1, vocab, dev, 1)
    out["pairs_round1"] = info1
    print("pairs 1", info1, flush=True)
    round1 = {}
    for b in BETAS:
        pol, secs = train(base.state_dict(), base, pp, win, lose, b, dev)
        round1[b] = pol
        row(f"DPO round 1, beta={b}", pol, {"beta": b, "train_seconds": secs})
    sel = max(BETAS, key=lambda b: next(r for r in out["rows"] if r["method"] == f"DPO round 1, beta={b}")["select_reward"])
    out["selected_beta"] = sel
    pol1 = round1[sel]
    for p in pol1.parameters():
        p.requires_grad_(False)
    pp2, win2, lose2, info2 = make_pairs(pol1, train_p2, vocab, dev, 3)
    out["pairs_round2"] = info2
    print("pairs 2", info2, flush=True)
    pol2, secs = train(pol1.state_dict(), pol1, pp2, win2, lose2, sel, dev)
    row(f"DPO round 2 (on-policy pairs, reference = round 1), beta={sel}", pol2, {"beta": sel, "train_seconds": secs})
    torch.manual_seed(5)
    demo = torch.tensor([TOK.encode("The pipeline writes the results ")], device=dev)[:, :32]
    out["examples"] = {n: TOK.decode(sample(m, demo, C)[0].tolist()) for n, m in
                       (("reference", base), ("round 1", pol1), ("round 2", pol2))}
    save("dpo_iter", out)


if __name__ == "__main__":
    main()
