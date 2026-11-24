# X-Ray Lab — team guide

**One sentence:** we do NOT diagnose disease; we explain how a computer detects a change (a bright round spot) in an image, step by step.

## 0. Mechanism, input, output (Step 1)
**Mechanism:** how a computer finds a bright spot in an image by comparing each pixel with its neighbours (local contrast + threshold + shape rules).
**Input (user changes it):** window size R, threshold T, spot brightness, noise level, which X-ray.
**Output:** the flagged spots, a verdict image, and a plain-English reason for each region.
**"Why this result?":** the green *Why?* box and the evidence bars update on every slider move.
**Honesty:** no AI model and no external service. All maths is our own code (`core.js`), computed live. Even the photon animation is computed from random numbers, not a prepared video.

## 1. Who explains which code (write your name in the last column)
| Step | Plain meaning | JavaScript (`core.js`) | Python (`xray_core.py`) | Explained by |
|---|---|---|---|---|
| Make the image | Draw a fake X-ray from numbers | `makeXray` | `make_xray` | |
| Average | "What is normal around here?" | `box` | `box` | |
| Lungs only | Search only the dark lung area | `lungs` | `lungs` | |
| Group pixels | Touching bright pixels = one blob | `regions` | `regions` | |
| Decide | Difference > T, then size and shape rules | `analyze` | `analyze` | |
| Score | Real spots found vs false alarms | `calc` (in `index.html`) | `score` | |
| Image file | Run the algorithm on a PNG/JPG | n/a (browser makes its own image) | `xray_lab.py` | |
| Check | Does it behave correctly? | (via `run_js.js`) | `test_xray.py` | |
| JS = Python | Prove both give the same answer | `run_js.js` | `test_6` in `test_xray.py` | |

Rule: everyone must be able to explain their rows **without reading the code aloud** — say what goes in, what comes out, and why.

## 2. Python vs JavaScript, in 3 sentences
Same five steps, same settings, same seeded image generator, so both find the same spots (a test proves it).
Different tools: JavaScript runs live in the browser (sliders, animation); Python runs in the terminal with numpy and runs the tests.
They do not call each other; the website does not run the Python file.

## 3. Why threshold changes what you see
A spot is kept only if (its brightness − its neighbourhood average) is above **T**.
Too high: real spots fall below T → missed. Too low: faint bumps pass (false alarms) and spots merge with ribs into one long region → rejected. See "When the algorithm fails" on the page; each case is reproduced by a test.

## 4. First-time viewer test (2 minutes)
Give a classmate who has never seen the project the page, say nothing, and time them.
1. Task: "Find the spot with the algorithm, then make it disappear." (pass: they move T)
2. Ask: "Does this tool diagnose illness?" (pass: **no**, it is a teaching simulation)
3. Ask: "Why did a spot disappear when T went up?" (pass: its brightness difference fell below T)
4. Ask: "What is the main idea in one sentence?" (pass: compares each pixel with its surroundings)
Pass = 3 of 4 right within 2 minutes. Note where they hesitated and fix that part of the page.

## 5. Likely judge questions
- *Is it AI?* No. It is hand-written rules, which is why every step can be explained.
- *Does it work on real X-rays?* Not tested. Our images are synthetic.
- *Why does it make mistakes?* It only sees brightness and shape. Noise and ribs can imitate a spot.
