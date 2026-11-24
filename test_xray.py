"""Key tests. Run: python3 -m unittest -v"""
import importlib.util, json, shutil, subprocess, tempfile, unittest
from pathlib import Path
import numpy as np
from xray_core import make_xray, analyze, score, CASES, CFG, W, H

def flagged(g, T=28):
    return [b for b in analyze(g, 14, T)["blobs"] if b["v"]]

class KeyTests(unittest.TestCase):
    def test_1_clear_spots_are_found(self):
        for seed, n in CASES:
            g, truth = make_xray(seed, n)
            self.assertEqual(score(analyze(g, 14, 28)["blobs"], truth), (n, 0))

    def test_2_healthy_image_gives_no_flags(self):
        self.assertEqual(flagged(make_xray(21, 0)[0]), [])

    def test_3_weak_spot_is_missed_honest_limit(self):
        g, truth = make_xray(7, 1, gain=8)
        self.assertEqual(score(analyze(g, 14, 28)["blobs"], truth), (0, 0))

    def test_4_fake_rib_streak_is_rejected(self):
        g, _ = make_xray(21, 0)
        y, x = np.mgrid[0:H, 0:W]
        g = np.clip(g + 60 * np.exp(-(((x - 100) / 30) ** 2 + ((y - 200) / 3) ** 2)), 0, 255).astype(np.float32)
        self.assertEqual(flagged(g), [])

    def test_5_threshold_trade_off(self):
        g, truth = make_xray(7, 1, gain=16)
        hits = {t: score(analyze(g, 14, t)["blobs"], truth)[0] for t in (8, 20, 28)}
        self.assertEqual(hits, {8: 0, 20: 1, 28: 0})   # too high misses, too low merges


    def test_7_the_three_failure_demos_on_the_website_are_real(self):
        g, t = make_xray(*CASES[1]); self.assertEqual(score(analyze(g, 14, 24)["blobs"], t), (0, 1))   # false alarm
        g, t = make_xray(*CASES[2]); self.assertEqual(score(analyze(g, 14, 44)["blobs"], t)[0], 0)      # missed spots
        g, t = make_xray(*CASES[0]); A = analyze(g, 14, 8)                                              # merged region
        self.assertEqual(score(A["blobs"], t)[0], 0)
        near = [b for b in A["blobs"] if np.hypot(b["x"] - t[0]["x"], b["y"] - t[0]["y"]) < 25]
        self.assertTrue(near and all(not b["v"] and b["a"] > 300 for b in near))   # spot merged into a big, rejected region

    def test_8_many_seeds_and_noise_levels(self):
        for seed in range(1, 7):
            for noise in (0, 14, 22):
                self.assertEqual(flagged(make_xray(seed, 0, noise=noise)[0]), [], f"healthy seed={seed} noise={noise}")
                g, t = make_xray(seed, 1, noise=noise)
                self.assertEqual(score(analyze(g, 14, 28)["blobs"], t)[1], 0, f"false alarm seed={seed} noise={noise}")

    @unittest.skipUnless(shutil.which("node"), "needs Node.js")
    def test_6_javascript_and_python_agree(self):
        """Same image, same R and T: candidate pixels, blob count and every flagged spot must match."""
        js = json.loads(subprocess.check_output(["node", str(Path(__file__).with_name("run_js.js"))]))
        self.assertEqual(len(js), 12)
        for c in js:
            g, _ = make_xray(c["seed"], c["n"]); A = analyze(g, 14, c["T"]); label = f"seed={c['seed']} T={c['T']}"
            self.assertEqual(int(A["c"].sum()), c["pixels"], label)
            self.assertEqual(len(A["blobs"]), c["blobs"], label)
            py = sorted((b for b in A["blobs"] if b["v"]), key=lambda b: (b["x"], b["y"]))
            jj = sorted(c["flagged"], key=lambda b: (b["x"], b["y"]))
            self.assertEqual(len(py), len(jj), label)
            for p, q in zip(py, jj):
                self.assertAlmostEqual(p["x"], q["x"], delta=0.2); self.assertAlmostEqual(p["y"], q["y"], delta=0.2)
                self.assertEqual(p["a"], q["a"], label)

    def test_9_website_runs_exactly_the_tested_javascript(self):
        core = Path(__file__).with_name("core.js").read_text(encoding="utf-8").split("// Lets Node")[0].split("if (typeof module")[0].strip()
        page = Path(__file__).with_name("index.html").read_text(encoding="utf-8")
        self.assertIn(core, page)

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "needs Pillow")
    def test_10_image_file_and_other_image_sizes(self):
        from PIL import Image
        g, truth = make_xray(7, 1)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "xray.png"; Image.fromarray(np.round(g).astype(np.uint8)).save(p)
            png = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
        self.assertEqual(score(analyze(png, 14, 28)["blobs"], truth), (1, 0))          # a PNG file works
        big = np.pad(png, ((20, 30), (25, 15)))                                          # 340 x 340 canvas, different size
        shifted = [dict(x=truth[0]["x"] + 25, y=truth[0]["y"] + 20)]
        self.assertEqual(score(analyze(big, 14, 28)["blobs"], shifted), (1, 0))

if __name__ == "__main__":
    unittest.main(verbosity=2)
