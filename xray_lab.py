"""X-Ray Lab - run the algorithm on an image FILE (educational, NOT a medical tool).

Usage:  python3 xray_lab.py image.png --window 14 --threshold 28 [--out diff.png]
        python3 xray_lab.py --demo 7          (analyse the website's synthetic image, seed 7)
Needs:  pip install numpy pillow

The algorithm itself lives in xray_core.py, the same steps as core.js on the website.
This file only loads an image and prints the result.
"""
import argparse
import numpy as np
from xray_core import analyze, make_xray


def load_gray(path):
    from PIL import Image                      # imported here so tests without Pillow still run
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--demo", type=int, help="use the synthetic image with this seed instead of a file")
    ap.add_argument("--window", type=int, default=14)
    ap.add_argument("--threshold", type=float, default=28)
    ap.add_argument("--out", help="save the difference map as a PNG")
    a = ap.parse_args()
    if a.demo is None and not a.image:
        ap.error("give an image file or --demo SEED")

    gray = make_xray(a.demo, 1)[0] if a.demo is not None else load_gray(a.image)
    res = analyze(gray, a.window, a.threshold)

    shown = [b for b in res["blobs"] if b["a"] >= 15]       # tiny specks are not listed
    for n, b in enumerate(shown, 1):
        tag = "FLAGGED " if b["v"] else "rejected"
        print(f"Region {n}: {tag} at ({b['x']:.0f},{b['y']:.0f}) brightness +{b['diff']:.0f}, size {b['a']}px, {b['why']}")
    if not shown:
        print("Nothing passed the threshold. Try a lower --threshold.")
    if a.out:
        from PIL import Image
        Image.fromarray(np.clip(128 + res["d"] * 4, 0, 255).astype(np.uint8)).save(a.out)
    print("Educational simulation only - not a medical diagnosis.")
