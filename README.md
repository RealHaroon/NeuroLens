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

### Tests

```bash
npm run test:parity    # TF.js vs Python: predictions + every activation
npm run test:pipeline  # activation probe, tensor-leak check, preprocessing robustness
npm run test:e2e       # production build in headless Chrome: draw, upload, predict
npm test               # all of the above
```

`test:e2e` uses your installed Chrome (`/usr/bin/google-chrome`, override with `CHROME_PATH`).
