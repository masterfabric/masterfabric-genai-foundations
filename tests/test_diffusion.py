

def test_guidance_scales_combine_conditional_and_unconditional():
    import torch

    from gaf.diffusion import CondUNet, guided
    torch.manual_seed(0)
    m = CondUNet(n_classes=4, ch=8, temb=16).eval()
    x, t, y = torch.randn(3, 1, 16, 16), torch.tensor([10, 500, 900]), torch.tensor([0, 1, 3])
    with torch.no_grad():
        ec, eu = m(x, t, y), m(x, t, None)
        assert torch.allclose(guided(m, y, 0.0)(x, t), eu, atol=1e-5)
        assert torch.allclose(guided(m, y, 1.0)(x, t), ec, atol=1e-5)
        assert torch.allclose(guided(m, y, 3.0)(x, t), eu + 3.0 * (ec - eu), atol=1e-4)
