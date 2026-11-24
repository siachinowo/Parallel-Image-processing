import sys, os, random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIZES = [256, 512, 1024, 2048, 4096]
random.seed(42); np.random.seed(42)

def synthetic(n=4096):
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    base = np.stack([
        127 + 100*np.sin(xx/300.0) * np.cos(yy/500.0),
        127 + 100*np.sin(yy/250.0 + xx/700.0),
        127 + 100*np.cos((xx+yy)/400.0)], axis=-1)
    base += np.random.normal(0, 6, base.shape)
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for _ in range(400):
        x, y = random.randint(0, n), random.randint(0, n)
        r = random.randint(20, 300)
        col = tuple(random.randint(0, 255) for _ in range(3))
        if random.random() < 0.5:
            d.ellipse((x, y, x+r, y+r), fill=col, outline=(0, 0, 0), width=3)
        else:
            d.rectangle((x, y, x+r, y+r//2), fill=col, outline=(255, 255, 255), width=3)
    for _ in range(150):
        d.line([random.randint(0, n) for _ in range(4)], fill=(0, 0, 0), width=random.randint(2, 8))
    return img

if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    src = Image.open(sys.argv[1]).convert("RGB")
    print("Using your photo:", sys.argv[1])
else:
    print("No photo given, generating a synthetic image...")
    src = synthetic()
W, H = src.size

for s in SIZES:
    out = f"data/{s}"
    os.makedirs(out, exist_ok=True)
    n_files = 100 if s <= 1024 else 20
    for i in range(n_files):
        crop = random.uniform(0.5, 1.0)
        cw, ch = int(W * crop), int(H * crop)
        x, y = random.randint(0, W - cw), random.randint(0, H - ch)
        img = src.crop((x, y, x + cw, y + ch)).resize((s, s), Image.LANCZOS)
        img.save(f"{out}/img_{i:03d}.png")
    print("done", s, flush=True)
