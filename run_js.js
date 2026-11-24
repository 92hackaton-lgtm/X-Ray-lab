// Prints what the JavaScript algorithm finds, as JSON. test_xray.py compares it with Python.
const C = require("./core.js");
const out = [];
for (const [seed, n] of C.CASES) for (const T of [16, 28, 40]) {
  const { g } = C.makeXray(seed, n), A = C.analyze(g, 14, T);
  out.push({ seed, n, T, pixels: A.c.reduce((a, b) => a + b, 0), blobs: A.blobs.length,
    flagged: A.blobs.filter(b => b.v).map(b => ({ x: +b.x.toFixed(1), y: +b.y.toFixed(1), a: b.a })) });
}
console.log(JSON.stringify(out));
