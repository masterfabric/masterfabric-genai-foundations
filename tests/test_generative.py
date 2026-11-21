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
