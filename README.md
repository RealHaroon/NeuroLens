# NeuroLens
3D CNN activation visualization tool.

A 3D, interactive visualizer of a small CNN's full forward pass on hand-drawn MNIST digits.
Runs entirely in the browser (TensorFlow.js + Three.js), no backend.

## Training (Phase 1)

Requires Python **3.12** (TensorFlow has no stable wheel for 3.13+ yet).

```bash
cd training
python3.12 -m venv .venv            # or: uv venv -p 3.12 .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python train.py           # ~40 s on CPU, ~98.4% test accuracy
.venv/bin/python export_weights.py  # -> web/public/model/model.json
```

The official `tensorflowjs` converter is not used: the only version that installs next to
TF 2.21 is 3.18.0, which predates Keras 3. `export_weights.py` instead writes the
architecture and all weights (float32, base64) to JSON, and `web/src/model.js` rebuilds the
identical model with `tf.layers` and loads the weights.

## Web app (Phase 2: inference and activations)

```bash
cd web
npm install
npm run dev      # local dev server
npm run build    # static production build in web/dist
```

- `src/model.js` rebuilds the Keras model from `public/model/model.json`.
- `src/activations.js` wraps it in a multi-output model so **one forward pass returns all 12
  stages** (`input, conv1, relu1, pool1, conv2, relu2, pool2, flatten, dense1, relu3, dense2,
  softmax`) as plain `Float32Array`s, and exposes the conv kernels `[3,3,Cin,Cout]` and dense
  matrices `[in,out]`. All tensors are disposed after each pass.
- `src/preprocess.js` does MNIST-style preprocessing: grayscale, auto-invert when the
  background is light, crop to the ink, area-resample into a 20×20 box, center by mass in a
  28×28 frame. The 28×28 result is shown in the UI.
- Only `tfjs-core`, `tfjs-layers` and the CPU backend are bundled. The model is small enough
  that CPU inference takes a few ms, and it keeps the GPU free for Three.js.

## 3D view (Phase 3)

`src/viz/` renders the forward pass with Three.js, left to right along X:

| Station | What you see |
|---|---|
| Input | 28×28 boxes, brightness = pixel value |
| Conv1 | 8 kernels (3×3, blue − / white 0 / red +) beside their 28×28 maps (before ReLU, blue − / red +) |
| ReLU + Pool1 | 14×14×8, aligned with their conv maps; negatives are gone |
| Conv2 | 16 kernels (3×3×8, as 8 slices) beside 14×14 maps |
| ReLU + Pool2 | 7×7×16 |
| Flatten | 784-cell column; **Unroll flatten** animates the 7×7×16 block into it |
| Dense | 64 spheres, size/brightness = activation after ReLU |
| Output | 10 spheres + softmax bars, prediction highlighted |

- Cells extrude toward the camera by |activation|, so tilted views read as a relief map.
- Only the top-k incoming connections of each dense neuron are drawn, ranked by |weight × input|,
  colored by sign, opacity by magnitude (k slider, default 8).
- Hover any cell or neuron for its exact value and incoming connections (conv receptive field,
  pool window, flatten source, dense top-k); click to pin, Esc to clear, double-click to focus.
- Performance: one `InstancedMesh` per layer (14 draw calls for the whole network), new inputs
  only rewrite instance colors/matrices, picking is analytic (ray–plane, no per-instance
  raycast), and frames render only when something changes. Orbiting holds 60 fps on Intel UHD
  integrated graphics.

### Tests

```bash
npm run test:parity    # TF.js vs Python: predictions + every activation
npm run test:pipeline  # activation probe, tensor-leak check, preprocessing robustness
npm run test:scene     # 3D data mapping: layout coverage, receptive fields, top-k
npm run test:e2e       # production build in headless Chrome: draw, upload, 3D hover, fps
npm test               # all of the above
```

`test:e2e` uses your installed Chrome (`/usr/bin/google-chrome`, override with `CHROME_PATH`) on
the real GPU when available.

> The npm scripts call `node node_modules/vite/bin/vite.js` instead of `vite` so they also work
> when the project path contains a `:` (e.g. `/run/media/…/Local Disk (D:)/…`), which breaks
> npm's `PATH`-based lookup of `node_modules/.bin`.
