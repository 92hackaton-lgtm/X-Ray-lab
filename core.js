/* ============================================================
   X-RAY LAB · core.js  — the whole algorithm, no drawing code.
   Same steps exist in xray_core.py (Python). Tests check they agree.

   PIPELINE (explain this to the judges in 5 lines):
     0. ROI      keep only the dark, air-filled lung area
     1. smooth   5x5 average of each pixel          (kills noise)
     2. average  (2R+1) window average              (what is "normal here")
     3. diff     smooth - average                   (brighter than neighbours?)
     4. T        diff > T inside ROI -> candidate pixel
     5. blobs    group touching candidates, then judge size and shape
   ============================================================ */

const W = 300, H = 360;                 // image size in pixels
const CASES = [[7, 1], [21, 0], [33, 2], [58, 1]];   // [seed, number of real spots]

// Every "magic number" has a name, so each one can be explained.
const CFG = {
  SMOOTH_R: 2,        // step 1: 5x5 window
  MIN_AREA: 15,       // smaller than this = noise
  MAX_AREA: 600,      // bigger than this = not a spot
  MAX_ASPECT: 1.8,    // long side / short side; ribs are much longer than 1.8
  MIN_FILL: 0.5,      // blob area / bounding box area (irregular blobs fail)
  MATCH_DIST: 16,     // a flag within 16px of a real spot counts as a hit
};

/* ---------- tiny random generator: same seed -> same image ---------- */
function rng(s) { return () => (s = (s * 16807) % 2147483647) / 2147483647 }
const G = (x, y, cx, cy, sx, sy) => Math.exp(-(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2));  // soft round bump

/* ---------- SYNTHETIC X-RAY (made by code, not a real patient) ----------
   opts.gain  = how bright the real spots are (default 42)
   opts.noise = detector noise strength        (default 14)            */
function makeXray(seed, n, opts = {}) {
  const gain = opts.gain ?? 42, noise = opts.noise ?? 14;
  const r = rng(seed), g = new Float32Array(W * H), truth = [], decoys = [];
  for (let i = 0; i < n; i++) truth.push({ x: (i % 2 ? 198 : 102) + (r() - .5) * 34, y: 135 + r() * 95, r: 5 + r() * 3 });
  for (let i = 0; i < 2; i++) decoys.push({ x: (i ? 198 : 102) + (r() - .5) * 36, y: 125 + r() * 110 });   // faint fake spots
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const m = 1 - ((x - 150) / 122) ** 2 - ((y - 190) / 176) ** 2;          // m>0 = inside the body ellipse
    let v = m > 0 ? 25 + 70 * Math.min(1, m * 4) : 12;                      // soft tissue / air outside
    if (m > 0) {
      v -= 48 * Math.exp(-((((x - 102) / 50) ** 4) + (((y - 190) / 88) ** 4)))      // left lung: darker
         + 48 * Math.exp(-((((x - 198) / 50) ** 4) + (((y - 190) / 88) ** 4)));     // right lung: darker
      v += 24 * G(x, y, 172, 238, 40, 46) + 70 * Math.exp(-(((x - 150) / 9) ** 2)); // heart, spine
      if (Math.abs(x - 150) > 12) for (let i = 0; i < 8; i++) {                      // 8 curved ribs
        const yy = 75 + i * 26 + (x - 150) ** 2 * .0045;
        v += 34 * Math.exp(-(((y - yy) / 4) ** 2)) * Math.min(1, m * 5);
      }
      for (const t of truth) v += gain * G(x, y, t.x, t.y, t.r, t.r);                // REAL spots
      for (const t of decoys) v += 13 * G(x, y, t.x, t.y, 4, 4);                     // faint decoys
    }
    g[y * W + x] = Math.max(0, Math.min(255, v + (r() - .5) * noise));               // add noise, clamp 0..255
  }
  return { g, truth };
}

/* ---------- box average of radius r (fast, via an "integral image") ---------- */
function box(src, r) {
  const S = new Float64Array((W + 1) * (H + 1)), o = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
    S[(y + 1) * (W + 1) + x + 1] = src[y * W + x] + S[y * (W + 1) + x + 1] + S[(y + 1) * (W + 1) + x] - S[y * (W + 1) + x];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const x0 = Math.max(0, x - r), x1 = Math.min(W, x + r + 1), y0 = Math.max(0, y - r), y1 = Math.min(H, y + r + 1);
    o[y * W + x] = (S[y1 * (W + 1) + x1] - S[y0 * (W + 1) + x1] - S[y1 * (W + 1) + x0] + S[y0 * (W + 1) + x0]) / ((x1 - x0) * (y1 - y0));
  }
  return o;
}

/* ---------- group touching pixels (flood fill). mask -> list of pixel-index lists ---------- */
function regions(mask) {
  const seen = new Uint8Array(W * H), out = [];
  for (let s = 0; s < W * H; s++) {
    if (seen[s] || !mask[s]) continue;
    const q = [s], px = []; seen[s] = 1;
    while (q.length) {
      const p = q.pop(), x = p % W; px.push(p);
      for (const k of [p - 1, p + 1, p - W, p + W]) {
        if (k < 0 || k >= W * H || seen[k] || !mask[k] || Math.abs(k % W - x) > 1) continue;
        seen[k] = 1; q.push(k);
      }
    }
    out.push(px);
  }
  return out;
}

/* ---------- STEP 0: find the lungs = the two biggest dark regions inside the body ---------- */
const ROIC = new WeakMap();
function lungs(g) {
  if (ROIC.has(g)) return ROIC.get(g);
  const rs = box(g, 10), bd = box(g, 3), bm = new Float32Array(W * H), m = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) bm[i] = bd[i] > 30 ? 1 : 0;           // body = not black
  const core = box(bm, 20);                                              // shrink body so skin edge is ignored
  for (let i = 0; i < W * H; i++) m[i] = core[i] > .999 && rs[i] < 70 ? 1 : 0;   // dark AND inside body
  const top = regions(m).sort((a, b) => b.length - a.length).slice(0, 2), roi = new Uint8Array(W * H);
  for (const px of top) for (const p of px) roi[p] = 1;
  const dl = box(roi, 8), dm = new Float32Array(W * H);                  // "closing": fill small holes
  for (let i = 0; i < W * H; i++) dm[i] = dl[i] > 0 ? 1 : 0;
  const er = box(dm, 8);
  for (let i = 0; i < W * H; i++) roi[i] = er[i] > .999 ? 1 : 0;
  ROIC.set(g, roi); return roi;
}

/* ---------- STEPS 1-5: the detector. Returns every blob with a plain-English reason. ---------- */
function analyze(g, R, T) {
  const sm = box(g, CFG.SMOOTH_R), bg = box(g, R), roi = lungs(g), d = new Float32Array(W * H), c = new Uint8Array(W * H), blobs = [];
  for (let i = 0; i < W * H; i++) { d[i] = sm[i] - bg[i]; c[i] = d[i] > T && roi[i] ? 1 : 0; }   // steps 3-4
  for (const px of regions(c)) {                                                                  // step 5
    let sx = 0, sy = 0, sd = 0, x0 = W, x1 = 0, y0 = H, y1 = 0;
    for (const p of px) { const x = p % W, y = (p / W) | 0; sx += x; sy += y; sd += d[p]; x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    const a = px.length, w = x1 - x0 + 1, h = y1 - y0 + 1, asp = Math.max(w, h) / Math.min(w, h), fill = a / (w * h);
    let v = 0, why;
    if (a < CFG.MIN_AREA) why = "too small → noise";
    else if (asp > CFG.MAX_ASPECT) why = "long & thin (ratio " + asp.toFixed(1) + ") → rib or edge";
    else if (a > CFG.MAX_AREA || fill < CFG.MIN_FILL) why = "too big / irregular → not a spot";
    else { v = 1; why = "round, compact, bright → suspicious"; }
    const df = sd / a;
    blobs.push({ x: sx / a, y: sy / a, a, w, h, v, why, diff: df,
      ev: [Math.min(1, df / (2 * T)), Math.max(0, 1 - (asp - 1) / 1.5), a >= CFG.MIN_AREA && a <= CFG.MAX_AREA ? 1 : .3] });   // evidence bars
  }
  return { bg, d, c, blobs };
}


// Lets Node (run_js.js) load this file; in the browser `module` does not exist, so this line is skipped.
if (typeof module !== "undefined") module.exports = { W, H, CASES, CFG, makeXray, box, regions, lungs, analyze };
