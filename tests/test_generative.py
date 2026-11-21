import math

import torch

from gaf.gan import gradient_penalty


def test_gradient_penalty_is_zero_for_a_1_lipschitz_linear_critic():
    critic = torch.nn.Linear(2, 1)
    with torch.no_grad():
        critic.weight.copy_(torch.tensor([[0.6, 0.8]]))  # gradient norm exactly 1 everywhere
    g = torch.Generator().manual_seed(0)
    assert gradient_penalty(critic, torch.randn(64, 2), torch.randn(64, 2), g).item() < 1e-10


def test_gradient_penalty_matches_hand_value():
    critic = torch.nn.Linear(2, 1)
    with torch.no_grad():
        critic.weight.copy_(torch.tensor([[3.0, 4.0]]))  # norm 5 -> (5 - 1)^2 = 16
    g = torch.Generator().manual_seed(0)
    assert math.isclose(gradient_penalty(critic, torch.randn(8, 2), torch.randn(8, 2), g).item(), 16.0, rel_tol=1e-6)


def test_cosine_schedule_decreases_from_one():
    from gaf.diffusion import cosine_schedule
    a = cosine_schedule(1000)
    assert a[0] > 0.9999 and a[-1] < 1e-5 and bool((a[1:] < a[:-1]).all())


class _Oracle(torch.nn.Module):
    """The exact eps-predictor when every training example is the point c."""

    def __init__(self, dif, c):
        super().__init__()
        self.dif, self.c = dif, c

    def forward(self, x, t):
        a = self.dif.abar[t].view(-1, 1)
        return (x - a.sqrt() * self.c) / (1 - a).sqrt()


def test_samplers_recover_the_data_point_with_an_exact_denoiser():
    from gaf.diffusion import Diffusion
    dif = Diffusion(200)
    c = torch.tensor([[1.5, -0.5]])
    g = torch.Generator().manual_seed(0)
    for steps in (None, 50, 10):
        x = dif.sample(_Oracle(dif, c), (16, 2), steps, gen=g)
        assert (x - c).abs().max().item() < 1e-3


def test_shapes_are_in_range_and_balanced():
    from gaf.diffusion import shapes
    x, y = shapes(4000, torch.Generator().manual_seed(0))
    assert x.shape == (4000, 1, 16, 16) and x.min() >= -1 and x.max() <= 1
    assert torch.bincount(y, minlength=4).min() > 900


def test_dpo_loss_is_log2_when_policy_equals_reference():
    from gaf.dpo import dpo_loss
    lp = torch.tensor([-10.0, -3.0])
    loss, acc, margin = dpo_loss(lp, lp - 1, lp, lp - 1, beta=0.1)
    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-6) and margin.item() == 0


def test_dpo_loss_falls_when_chosen_gains_probability():
    from gaf.dpo import dpo_loss
    ref_w, ref_l = torch.tensor([-5.0]), torch.tensor([-5.0])
    lo, _, _ = dpo_loss(ref_w + 2, ref_l, ref_w, ref_l, beta=0.5)
    assert lo.item() < math.log(2)


def test_completion_logprob_matches_token_by_token():
    from gaf.dpo import completion_logprob
    from gaf.transformer import GPT, GPTConfig
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=11, block_size=16, n_layer=1, n_head=2, d_model=16, dropout=0.0)).eval()
    p, c = torch.randint(11, (2, 4)), torch.randint(11, (2, 3))
    manual = torch.zeros(2)
    seq = p
    for j in range(3):
        manual += m(seq)[0][:, -1].log_softmax(-1).gather(1, c[:, j:j + 1]).squeeze(1)
        seq = torch.cat([seq, c[:, j:j + 1]], 1)
    assert torch.allclose(completion_logprob(m, p, c), manual, atol=1e-5)


def test_reward_counts_known_words_and_ignores_the_cut_last_word():
    from gaf.dpo import reward
    v = {"the", "data", "pipeline"}
    assert reward("the data pipelxne wri", v) == 2 / 3
    assert reward("nospace", v) == 0.0
