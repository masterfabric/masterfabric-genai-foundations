"""Write the four notebooks from code cells kept here (reviewable as plain Python), then execute them.

Each notebook runs a small live demo of the from-scratch module and loads the measured results of
the full experiment from results/*.json. Executed outputs are committed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
SETUP = "import sys, json\nsys.path.insert(0, '..')\nimport torch\nfrom gaf.common import device\ndev = device(); print(torch.__version__, dev)"

NB = {
    "01_attention_and_gpt.ipynb": [
        ("md", "# 01 · Attention and a GPT from scratch\nScaled dot-product attention written out, checked against PyTorch's fused kernel, then the pretrained character model generates text."),
        ("code", SETUP),
        ("code", "import torch.nn.functional as F\nfrom gaf.transformer import attention\ntorch.manual_seed(0)\nq, k, v = (torch.randn(1, 2, 6, 4) for _ in range(3))\nours = attention(q, k, v)\nprint('max |ours - torch SDPA| =', (ours - F.scaled_dot_product_attention(q, k, v, is_causal=True)).abs().max().item())"),
        ("code", "# the causal mask: each position attends only to itself and the past\nimport math\ns = (q @ k.transpose(-2, -1) / math.sqrt(4))[0, 0]\nmask = torch.ones(6, 6, dtype=torch.bool).tril()\nprint(s.masked_fill(~mask, float('-inf')).softmax(-1).round(decimals=2))"),
        ("code", "from pathlib import Path\nfrom gaf.transformer import GPT, GPTConfig\nfrom experiments.train_gpt import TOK\nck = Path('../checkpoints/gpt_pretrain.pt')\nif ck.exists():\n    state = torch.load(ck, map_location='cpu')\n    m = GPT(GPTConfig(**state['cfg'])); m.load_state_dict(state['model']); m.eval()\n    torch.manual_seed(3)\n    print(TOK.decode(m.generate(torch.tensor([TOK.encode('## Context\\n\\n')]), 300)[0].tolist()))\nelse:\n    print('run experiments/train_gpt.py first')"),
        ("code", "r = json.load(open('../results/gpt_pretrain.json'))\nprint({k: r[k] for k in ('params', 'seconds', 'tokens_per_s', 'uniform_loss', 'best_val', 'best_step')})"),
    ],
    "02_lora.ipynb": [
        ("md", "# 02 · LoRA from scratch\nA frozen linear layer plus a low-rank update B·A. The adapter starts as a no-op and merges back exactly."),
        ("code", SETUP),
        ("code", "from gaf.lora import LoRALinear\ntorch.manual_seed(0)\nbase = torch.nn.Linear(128, 128)\nl = LoRALinear(base, r=8, alpha=16)\nx = torch.randn(4, 128)\nprint('identity at init:', torch.allclose(l(x), base(x)))\nwith torch.no_grad(): l.B.normal_(std=0.02)\nprint('merged == adapter:', torch.allclose(l(x), l.merged()(x), atol=1e-5))\nprint('trainable', l.A.numel() + l.B.numel(), 'vs full', base.weight.numel() + base.bias.numel())"),
        ("code", "r = json.load(open('../results/lora.json'))\nfor run in r['runs']:\n    print(f\"{run['method']:18} trainable {run['trainable']:>8,}  val(new) {run['val_new']:.3f}  val(original) {run['val_original']:.3f}  {run['seconds']}s\")"),
    ],
    "03_gan.ipynb": [
        ("md", "# 03 · A GAN from scratch\nGenerator and discriminator MLPs on a ring of eight Gaussians, so mode coverage can be counted."),
        ("code", SETUP),
        ("code", "from gaf.gan import train\nr = train(seed=5, steps=1500, snapshots=(1500,))\nfor c in r['curve']: print(c)"),
        ("code", "import matplotlib.pyplot as plt, numpy as np\nfrom gaf.gan import MODES\np = np.array(r['snapshots'][1500])\nplt.figure(figsize=(4, 4)); plt.scatter(p[:, 0], p[:, 1], s=3, alpha=0.5); plt.scatter(MODES[:, 0], MODES[:, 1], marker='x', c='r'); plt.axis('equal'); plt.show()"),
        ("code", "g = json.load(open('../results/gan.json'))\nfor run in g['runs']: print(run['loss'], run['seed'], run['final'])"),
    ],
    "04_policy_gradient.ipynb": [
        ("md", "# 04 · REINFORCE on CartPole\nVectorised cart-pole dynamics in torch and a policy-gradient learner with an optional value baseline."),
        ("code", SETUP),
        ("code", "from gaf.rl import train\nr = train(seed=7, baseline=True, updates=40)\nprint([c['mean_return'] for c in r['curve']][::4])"),
        ("code", "g = json.load(open('../results/rl.json'))\nfor run in g['runs']: print('baseline' if run['baseline'] else 'plain', run['seed'], run['final_mean_return'], run['first_update_at_195'])"),
    ],
}


def main(execute: bool = True):
    sys.path.insert(0, str(HERE.parent))
    for name, cells in NB.items():
        nb = nbformat.v4.new_notebook()
        nb.cells = [nbformat.v4.new_markdown_cell(s) if t == "md" else nbformat.v4.new_code_cell(s) for t, s in cells]
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        if execute:
            NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
        nbformat.write(nb, HERE / name)
        print("wrote", name)


if __name__ == "__main__":
    main()
