import torch
import torch.nn.functional as F

from gaf.lora import LoRALinear, add_lora, trainable
from gaf.transformer import GPT, GPTConfig, attention


def test_attention_matches_torch_sdpa():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 16, 8) for _ in range(3))
    ours = attention(q, k, v, causal=True)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(ours, ref, atol=1e-5)


def test_causality_future_tokens_do_not_change_past_logits():
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=20, block_size=16, n_layer=2, n_head=2, d_model=32, dropout=0.0)).eval()
    x = torch.randint(0, 20, (1, 16))
    y = x.clone(); y[0, 10:] = (y[0, 10:] + 1) % 20
    assert torch.allclose(m(x)[0][:, :10], m(y)[0][:, :10], atol=1e-5)


def test_untrained_loss_is_close_to_uniform():
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=50, block_size=16, n_layer=2, n_head=2, d_model=32))
    _, loss = m(torch.randint(0, 50, (4, 16)), torch.randint(0, 50, (4, 16)))
    assert abs(loss.item() - torch.log(torch.tensor(50.0)).item()) < 0.2


def test_lora_starts_as_identity_and_merges_exactly():
    torch.manual_seed(0)
    base = torch.nn.Linear(16, 8)
    l = LoRALinear(base, r=4)
    x = torch.randn(3, 16)
    assert torch.allclose(l(x), base(x))
    with torch.no_grad():
        l.B.normal_()
    assert torch.allclose(l(x), l.merged()(x), atol=1e-5)


def test_add_lora_freezes_base():
    m = GPT(GPTConfig(vocab_size=30, block_size=16, n_layer=2, n_head=2, d_model=32))
    total = sum(p.numel() for p in m.parameters())
    add_lora(m, r=4)
    assert 0 < trainable(m) < total * 0.2
