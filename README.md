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

Parity check (runs the browser loader in Node and compares every layer against Python):

```bash
cd web && npm install && npm run test:parity
```
