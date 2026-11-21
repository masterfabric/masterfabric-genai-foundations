# 0001. What "from scratch" means here

- Status: Accepted
- Date: 2026-11-21

## Decision

Built by hand: attention (QKᵀ/√d, causal mask, softmax, weighted V), multi-head split, transformer
blocks, sampling, LoRA adapters and their merge, the GAN training loop and both generator losses,
CartPole dynamics, and REINFORCE with a value baseline.

Taken from PyTorch: tensors, autograd, `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, activations,
cross-entropy and the Adam/AdamW optimisers. Re-deriving autograd or optimisers would add code without
adding understanding of generative models.

No Hugging Face, Gym or other model library is used. Fused attention is used only as the reference
in a test: our attention matches `scaled_dot_product_attention` within 1.5e-7.

## Consequences

- The fused kernel would be faster; the explicit version is kept for readability.
- Results are toy-scale by design: a 0.82M-parameter character model, a 2-D GAN, CartPole.
