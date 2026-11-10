"""Render an animated GitHub avatar: a rotating indigo/cyan ring around a monogram or a photo.

    python3 avatar/make_avatar.py                    # monogram "PS"
    python3 avatar/make_avatar.py --photo me.jpg     # your photo inside the ring

Writes avatar/avatar.gif (500x500, looping, under GitHub's 1 MB avatar limit).
"""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SIZE = 500
SS = 2  # supersampling factor for smooth edges
FRAMES = 36
FRAME_MS = 60
BG = (12, 15, 24)
STOPS = [(79, 70, 229), (8, 145, 178), (56, 189, 248), (139, 135, 255), (79, 70, 229)]
FONT = "/System/Library/Fonts/Avenir Next.ttc"


def conic(n, phase):
    """RGB array of a conic gradient through STOPS, rotated by phase (0..1)."""
    y, x = np.mgrid[0:n, 0:n] - (n - 1) / 2
    a = ((np.arctan2(y, x) / (2 * math.pi)) - phase) % 1.0
    pos = a * (len(STOPS) - 1)
    i = np.minimum(pos.astype(int), len(STOPS) - 2)
    f = (pos - i)[..., None]
    s = np.array(STOPS, dtype=float)
    return (s[i] * (1 - f) + s[i + 1] * f).astype(np.uint8)


def circle_mask(n, r_outer, r_inner=0.0):
    y, x = np.mgrid[0:n, 0:n] - (n - 1) / 2
    d = np.hypot(x, y)
    return ((d <= r_outer) & (d >= r_inner)).astype(np.uint8) * 255


def center_layer(n, r, photo):
    layer = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    if photo:
        img = Image.open(photo).convert("RGB")
        side = min(img.size)
        left, top = (img.width - side) // 2, (img.height - side) // 2
        img = img.crop((left, top, left + side, top + side)).resize((2 * r, 2 * r), Image.LANCZOS)
    else:
        # Dark radial disc with a glowing monogram
        yy, xx = np.mgrid[0 : 2 * r, 0 : 2 * r] - (r - 0.5)
        t = np.clip(np.hypot(xx, yy) / r, 0, 1)[..., None]
        inner, outer = np.array([30, 36, 58.0]), np.array(BG, dtype=float)
        img = Image.fromarray((inner * (1 - t) + outer * t).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(FONT, int(r * 0.9), index=0)  # Avenir Next Bold
        draw.text((r, r), "PS", font=font, fill=(238, 241, 248), anchor="mm")
    mask = Image.fromarray(circle_mask(2 * r, r))
    layer.paste(img, ((n - 2 * r) // 2, (n - 2 * r) // 2), mask)
    return layer


def frame(k, center):
    n = SIZE * SS
    phase = k / FRAMES
    r_out, width = int(n * 0.47), int(n * 0.045)
    canvas = Image.new("RGBA", (n, n), BG + (255,))
    grad = Image.fromarray(conic(n, phase)).convert("RGBA")

    # Soft glow under the ring, breathing over the loop
    glow_alpha = 150 + int(70 * math.sin(2 * math.pi * phase))
    glow_mask = Image.fromarray(circle_mask(n, r_out, r_out - width)).filter(ImageFilter.GaussianBlur(n * 0.025))
    glow_mask = glow_mask.point(lambda v: v * glow_alpha // 255)
    canvas.paste(grad, (0, 0), glow_mask)

    # Crisp ring
    canvas.paste(grad, (0, 0), Image.fromarray(circle_mask(n, r_out, r_out - width)))

    # Orbiting highlight dot riding the ring
    ang = 2 * math.pi * phase - math.pi / 2
    rc = r_out - width / 2
    cx, cy = n / 2 + rc * math.cos(ang), n / 2 + rc * math.sin(ang)
    dot = Image.new("L", (n, n), 0)
    ImageDraw.Draw(dot).ellipse([cx - width, cy - width, cx + width, cy + width], fill=255)
    canvas.paste((255, 255, 255, 255), (0, 0), dot.filter(ImageFilter.GaussianBlur(width * 0.35)))

    canvas.alpha_composite(center)
    return canvas.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", help="square-ish photo to place inside the ring")
    ap.add_argument("--out", default=str(Path(__file__).with_name("avatar.gif")))
    args = ap.parse_args()

    n = SIZE * SS
    center = center_layer(n, int(n * 0.47) - int(n * 0.045) - int(n * 0.02), args.photo)
    frames = [frame(k, center) for k in range(FRAMES)]

    # One shared palette across all frames prevents colour flicker
    strip = Image.new("RGB", (SIZE, SIZE * 4))
    for i, f in enumerate(frames[:: FRAMES // 4][:4]):
        strip.paste(f, (0, SIZE * i))
    palette = strip.quantize(colors=200, method=Image.Quantize.MEDIANCUT)
    out = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    out[0].save(args.out, save_all=True, append_images=out[1:], duration=FRAME_MS, loop=0, optimize=True, disposal=1)
    print(f"{args.out}: {Path(args.out).stat().st_size / 1024:.0f} KB, {FRAMES} frames")


if __name__ == "__main__":
    main()
