"""
X-RAY LAB · xray_core.py — the SAME algorithm as core.js, written in Python.

PIPELINE (identical to the website):
  0. ROI     keep only the dark, air-filled lung area
  1. smooth  5x5 average of each pixel
  2. average (2R+1) window average  ("what is normal here")
  3. diff    smooth - average      (brighter than its neighbours?)
  4. T       diff > T inside ROI -> candidate pixel
  5. blobs   group touching candidates, judge size and shape

Run it:   python3 xray_core.py          (prints the result for the 4 demo cases)
Test it:  python3 -m unittest -v        (see test_xray.py)

Educational simulation on SYNTHETIC images. Not a medical device.
"""
import numpy as np

W, H = 300, 360
CASES = [(7, 1), (21, 0), (33, 2), (58, 1)]          # (seed, number of real spots)
CFG = dict(SMOOTH_R=2, MIN_AREA=15, MAX_AREA=600, MAX_ASPECT=1.8, MIN_FILL=0.5, MATCH_DIST=16)


# ---------- same tiny random generator as the website ----------
def rng(seed):
    s = seed
    while True:
        s = (s * 16807) % 2147483647
        yield s / 2147483647


# ---------- SYNTHETIC X-RAY ----------
def make_xray(seed, n, gain=42, noise=14):
    r = rng(seed)
    truth = [dict(x=(198 if i % 2 else 102) + (next(r) - .5) * 34, y=135 + next(r) * 95, r=5 + next(r) * 3) for i in range(n)]
    decoys = [dict(x=(198 if i else 102) + (next(r) - .5) * 36, y=125 + next(r) * 110) for i in range(2)]
    noise_vals = np.array([next(r) for _ in range(W * H)]).reshape(H, W)   # one random number per pixel, row by row

    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    gauss = lambda cx, cy, sx, sy: np.exp(-(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))
    m = 1 - ((x - 150) / 122) ** 2 - ((y - 190) / 176) ** 2
    inside = m > 0
    v = np.where(inside, 25 + 70 * np.minimum(1, m * 4), 12.0)
    body = -48 * (np.exp(-(((x - 102) / 50) ** 4 + ((y - 190) / 88) ** 4))
                  + np.exp(-(((x - 198) / 50) ** 4 + ((y - 190) / 88) ** 4)))   # both lungs darker
    body += 24 * gauss(172, 238, 40, 46) + 70 * np.exp(-(((x - 150) / 9) ** 2))
    for i in range(8):                                                          # 8 curved ribs
        yy = 75 + i * 26 + (x - 150) ** 2 * .0045
        body += np.where(np.abs(x - 150) > 12, 34 * np.exp(-(((y - yy) / 4) ** 2)) * np.minimum(1, m * 5), 0)
    for t in truth:
        body += gain * gauss(t["x"], t["y"], t["r"], t["r"])
    for t in decoys:
        body += 13 * gauss(t["x"], t["y"], 4, 4)
    v = v + np.where(inside, body, 0)
    img = np.clip(v + (noise_vals - .5) * noise, 0, 255).astype(np.float32)
    return img, truth


# ---------- box average of radius r (integral image, like the website) ----------
def box(src, r):
    H, W = src.shape            # works for any image size, not only 300x360
    S = np.zeros((H + 1, W + 1))
    S[1:, 1:] = np.cumsum(np.cumsum(src.astype(np.float64), 0), 1)
    ys, xs = np.arange(H), np.arange(W)
    y0, y1 = np.maximum(0, ys - r), np.minimum(H, ys + r + 1)
    x0, x1 = np.maximum(0, xs - r), np.minimum(W, xs + r + 1)
    tot = S[np.ix_(y1, x1)] - S[np.ix_(y0, x1)] - S[np.ix_(y1, x0)] + S[np.ix_(y0, x0)]
    return (tot / np.outer(y1 - y0, x1 - x0)).astype(np.float32)


# ---------- group touching pixels (flood fill) -> list of pixel lists ----------
def regions(mask):
    H, W = mask.shape            # works for any image size, not only 300x360
    seen = np.zeros((H, W), bool)
    out = []
    for sy, sx in zip(*np.nonzero(mask)):
        if seen[sy, sx]:
            continue
        stack, px = [(sy, sx)], []
        seen[sy, sx] = True
        while stack:
            cy, cx = stack.pop()
            px.append((cy, cx))
            for ny, nx in ((cy, cx - 1), (cy, cx + 1), (cy - 1, cx), (cy + 1, cx)):
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        out.append(px)
    return out


# ---------- STEP 0: lungs = the two biggest dark regions inside the body ----------
def lungs(g):
    H, W = g.shape            # works for any image size, not only 300x360
    rs, bd = box(g, 10), box(g, 3)
    core = box((bd > 30).astype(np.float32), 20)          # body shrunk so the skin edge is ignored
    m = (core > .999) & (rs < 70)
    roi = np.zeros((H, W), np.float32)
    for px in sorted(regions(m), key=len, reverse=True)[:2]:
        for cy, cx in px:
            roi[cy, cx] = 1
    dm = (box(roi, 8) > 0).astype(np.float32)             # "closing": fill small holes
    return box(dm, 8) > .999


# ---------- STEPS 1-5: the detector ----------
def analyze(g, R, T):
    sm, bg, roi = box(g, CFG["SMOOTH_R"]), box(g, R), lungs(g)
    d = sm - bg
    cand = (d > T) & roi
    blobs = []
    for px in regions(cand):
        ys, xs = np.array([p[0] for p in px]), np.array([p[1] for p in px])
        a, w, h = len(px), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        asp, fill = max(w, h) / min(w, h), a / (w * h)
        if a < CFG["MIN_AREA"]:
            ok, why = False, "too small -> noise"
        elif asp > CFG["MAX_ASPECT"]:
            ok, why = False, f"long & thin (ratio {asp:.1f}) -> rib or edge"
        elif a > CFG["MAX_AREA"] or fill < CFG["MIN_FILL"]:
            ok, why = False, "too big / irregular -> not a spot"
        else:
            ok, why = True, "round, compact, bright -> suspicious"
        blobs.append(dict(x=xs.mean(), y=ys.mean(), a=a, w=int(w), h=int(h), v=ok, why=why, diff=float(d[ys, xs].mean())))
    return dict(bg=bg, d=d, c=cand, blobs=blobs)


def score(blobs, truth):
    """hits = real spots found, fa = false alarms (flagged blobs not near a real spot)."""
    sus = [b for b in blobs if b["v"]]
    near = lambda b, t: np.hypot(b["x"] - t["x"], b["y"] - t["y"]) < CFG["MATCH_DIST"]
    hit = sum(any(near(b, t) for b in sus) for t in truth)
    fa = sum(not any(near(b, t) for t in truth) for b in sus)
    return hit, fa


if __name__ == "__main__":
    for seed, n in CASES:
        g, truth = make_xray(seed, n)
        A = analyze(g, 14, 28)
        hit, fa = score(A["blobs"], truth)
        print(f"seed {seed}: real spots {n}, found {hit}, false alarms {fa}")
