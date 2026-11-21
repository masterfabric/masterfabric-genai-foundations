"""Overview diagram (Pillow): the four from-scratch building blocks and how they connect."""
from mfviz import GREEN, GREY, ICE, IMG, ORANGE, PURPLE, RED, WHITE, YELLOW, arrow, box, canvas, font


def overview():
    im, d = canvas(1600, 900)
    d.text((60, 40), "MasterFabric GenAI Foundations · what is built from scratch", font=font(34, 800), fill=WHITE)
    d.text((60, 82), "PyTorch tensors, autograd and optimisers only; every model trained on an Apple M4 Pro.", font=font(20, 450), fill=GREY)
    box(d, (50, 170, 520, 470), "Transformer (gaf/transformer.py)", ["scaled dot-product attention", "causal mask · multi-head split",
                                                                       "pre-norm blocks · GELU MLP", "weight tying · top-k sampling",
                                                                       "0.82M params, char-level"], ICE, line_size=18)
    box(d, (620, 170, 1090, 470), "LoRA (gaf/lora.py)", ["W x + (α/r) · B A x", "W frozen · B = 0 at init", "wraps q, k, v, o, MLP",
                                                          "exact merge for inference", "ranks 2 · 8 · 32 vs full FT"], PURPLE, line_size=18)
    box(d, (50, 560, 520, 840), "GAN (gaf/gan.py)", ["generator + discriminator MLPs", "8-Gaussian ring target",
                                                      "saturating vs non-saturating", "metric: modes covered"], GREEN, line_size=18)
    box(d, (620, 560, 1090, 840), "Policy gradient (gaf/rl.py)", ["CartPole dynamics in torch", "16 vectorised episodes",
                                                                   "REINFORCE ± value baseline", "metric: episodes to 195"], ORANGE, line_size=18)
    box(d, (1180, 170, 1550, 470), "Text corpora", ["repos 1-3 docs: pretrain", "   104k chars train", "repos 4-6 ADRs: adapt",
                                                     "   14k chars train"], YELLOW, line_size=18)
    box(d, (1180, 560, 1550, 840), "Evidence", ["tests/: SDPA match,", "   causality, LoRA merge", "results/*.json", "notebooks/ (executed)",
                                                 "docs/images/"], RED, line_size=18)
    arrow(d, (520, 320), (620, 320), ICE, "frozen base")
    d.line([(1365, 170), (1365, 135), (285, 135)], fill=YELLOW, width=3)
    arrow(d, (285, 135), (285, 170), YELLOW)
    d.text((700, 112), "pretrain", font=font(16, 600), fill=YELLOW)
    arrow(d, (1180, 380), (1090, 380), YELLOW, "adapt")
    arrow(d, (1090, 700), (1180, 700), GREY)
    im.save(IMG / "overview.png", optimize=True)


if __name__ == "__main__":
    overview()
    print("diagram written")
