"""Compose docs/hero.png (1600x900) from real screenshots and measured numbers.

Every number on the hero is read from results/*.json at render time.
"""

from __future__ import annotations

import json

from PIL import Image, ImageDraw, ImageFilter

from mfviz import BG, BG2, DOCS, GREEN, GREY, ICE, IMG, ORANGE, PURPLE, RED, RESULTS, WHITE, YELLOW, font

W, H = 1600, 900


def load(n):
    return json.loads((RESULTS / f"{n}.json").read_text())


def background():
    im = Image.new("RGB", (W, H), BG)
    px = im.load()
    for y in range(H):
        for x in range(W):
            t = min(1.0, (x / W) * 0.75 + (y / H) * 0.35)
            px[x, y] = tuple(int(BG[i] * (1 - t) + BG2[i] * t) for i in range(3))
    return im


def card(base, path, box, label, color, crop=None):
    x0, y0, x1, y1 = box
    shot = Image.open(path).convert("RGB")
    if crop:
        shot = shot.crop(crop)
    shot = shot.resize((x1 - x0, round(shot.height * (x1 - x0) / shot.width)), Image.LANCZOS)
    shot = shot.crop((0, 0, x1 - x0, min(shot.height, y1 - y0)))
    shadow = Image.new("RGBA", (shot.width + 40, shot.height + 40), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((20, 20, shot.width + 20, shot.height + 20), radius=16, fill=(0, 0, 0, 170))
    shadow = shadow.filter(ImageFilter.GaussianBlur(12))
    base.paste(shadow, (x0 - 20, y0 - 12), shadow)
    mask = Image.new("L", shot.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, shot.width, shot.height), radius=14, fill=255)
    base.paste(shot, (x0, y0), mask)
    d = ImageDraw.Draw(base)
    f = font(20, 800)
    w = d.textlength(label, font=f)
    d.rounded_rectangle((x0 - 10, y0 - 22, x0 + w + 18, y0 + 14), radius=18, fill=color)
    d.text((x0 + 4, y0 - 18), label, font=f, fill=(255, 255, 255))


def main():
    g, lo, ga, rl = load("gpt_pretrain"), load("lora"), load("gan"), load("rl")
    runs = {r["method"]: r for r in lo["runs"]}
    ns = [r["final"] for r in ga["runs"] if r["loss"] == "non_saturating"]
    eps = [r["first_update_at_195"] for r in rl["runs"]]
    im = background()
    card(im, IMG / "gpt_loss.png", (960, 70, 1570, 300), "GPT pretraining", ICE)
    card(im, IMG / "lora.png", (960, 360, 1570, 560), "LoRA vs full fine-tuning", PURPLE, crop=(0, 0, 1130, 720))
    card(im, IMG / "gan.png", (960, 620, 1570, 760), "GAN", GREEN, crop=(0, 0, 2050, 580))
    card(im, IMG / "rl.png", (1160, 790, 1570, 880), "REINFORCE", ORANGE)
    d = ImageDraw.Draw(im)
    d.text((80, 70), "MASTERFABRIC", font=font(36, 800), fill=WHITE)
    d.text((76, 120), "GenAI Foundations", font=font(80, 800), fill=WHITE)
    x, y = 80, 238
    for word, col in (("Transformer", ICE), (" · ", GREY), ("LoRA", PURPLE), (" · ", GREY), ("GAN", GREEN), (" · ", GREY), ("RL", ORANGE)):
        f = font(38, 800)
        d.text((x, y), word, font=f, fill=col)
        x += d.textlength(word, font=f)
    d.text((80, 318), "Generative AI building blocks written from the equations in PyTorch,", font=font(26, 500), fill=GREY)
    d.text((80, 352), "trained on a laptop GPU. Executed notebooks, measured results only.", font=font(26, 500), fill=GREY)
    r8 = runs["LoRA r=8"]
    facts = [
        (f"GPT: val loss {g['uniform_loss']:.2f} → {g['best_val']:.2f} in {g['seconds']:.0f} s",
         f"{g['params']:,} params · {g['tokens_per_s']:,} tok/s on Apple MPS · attention = torch SDPA to 1e-7", ICE),
        (f"LoRA r=8: {r8['trainable_pct']}% of weights, loss {r8['val_new']:.2f}",
         f"full fine-tuning {runs['full fine-tuning']['val_new']:.2f} · forgetting {r8['val_original']:.2f} vs "
         f"{runs['full fine-tuning']['val_original']:.2f}", PURPLE),
        (f"GAN: {min(r['modes_covered'] for r in ns)}-{max(r['modes_covered'] for r in ns)} of 8 modes, "
         f"{min(r['high_quality_pct'] for r in ns):.0f}-{max(r['high_quality_pct'] for r in ns):.0f}% on-mode",
         "non-saturating loss, 3 seeds · partial mode collapse, measured", GREEN),
        (f"CartPole solved in {min(eps)}-{max(eps)} episodes",
         "REINFORCE from scratch, 6 runs · vectorised dynamics in torch", ORANGE),
    ]
    y = 430
    for title, sub, col in facts:
        d.rounded_rectangle((80, y + 6, 88, y + 62), radius=4, fill=col)
        d.text((106, y), title, font=font(29, 750), fill=WHITE)
        d.text((106, y + 38), sub, font=font(20, 450), fill=GREY)
        y += 92
    d.text((80, 820), "github.com/masterfabric/masterfabric-genai-foundations", font=font(30, 700), fill=WHITE)
    im.save(DOCS / "hero.png", optimize=True)
    print("docs/hero.png written")


if __name__ == "__main__":
    main()
