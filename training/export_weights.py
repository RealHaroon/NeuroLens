"""Export the trained Keras model to a self-describing JSON file for the browser.

The `tensorflowjs` converter is not used: the only release that resolves next to
TensorFlow 2.21 is 3.18.0 (2022), which predates Keras 3 and cannot read `.keras`
files. Instead we dump the architecture plus every weight tensor and rebuild the
identical model in TF.js with the tf.layers API (see web/src/model.js).

Weights are stored as little-endian float32, base64-encoded, so they round-trip
bit-exactly while keeping the file small.

Also writes web/test/fixtures/parity.json: a few MNIST test images with the
Python model's outputs at every layer, used by web/test/parity.mjs.

Usage:  python export_weights.py
"""

import base64
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_PATH = HERE / "artifacts" / "mnist_cnn.keras"
SAMPLE_PATH = HERE / "artifacts" / "test_sample.npz"
OUT_MODEL = ROOT / "web" / "public" / "model" / "model.json"
OUT_PARITY = ROOT / "web" / "test" / "fixtures" / "parity.json"
PARITY_IMAGES = 20
LAYER_IMAGES = 2


def b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype="<f4").tobytes()).decode("ascii")


def tensor(arr: np.ndarray) -> dict:
    return {"shape": list(arr.shape), "dtype": "float32", "data": b64(arr)}


def layer_spec(layer) -> dict:
    cls = type(layer).__name__
    cfg = layer.get_config()
    if cls == "Conv2D":
        spec = {
            "type": "conv2d",
            "filters": cfg["filters"],
            "kernelSize": list(cfg["kernel_size"]),
            "strides": list(cfg["strides"]),
            "padding": cfg["padding"],
            "useBias": cfg["use_bias"],
        }
        assert cfg["activation"] == "linear", f"{layer.name}: activation must be a separate layer"
    elif cls == "Dense":
        spec = {"type": "dense", "units": cfg["units"], "useBias": cfg["use_bias"]}
        assert cfg["activation"] == "linear", f"{layer.name}: activation must be a separate layer"
    elif cls == "ReLU":
        spec = {"type": "relu"}
    elif cls == "MaxPooling2D":
        spec = {
            "type": "maxPooling2d",
            "poolSize": list(cfg["pool_size"]),
            "strides": list(cfg["strides"]),
            "padding": cfg["padding"],
        }
    elif cls == "Flatten":
        spec = {"type": "flatten"}
    elif cls == "Softmax":
        spec = {"type": "softmax"}
    else:
        raise ValueError(f"Unsupported layer type {cls} ({layer.name})")

    spec["name"] = layer.name
    spec["outputShape"] = list(layer.output.shape[1:])
    names = ["kernel", "bias"]
    spec["weights"] = {names[i]: tensor(w) for i, w in enumerate(layer.get_weights())}
    return spec


def main():
    model = keras.models.load_model(MODEL_PATH)
    sample = np.load(SAMPLE_PATH)
    x, y = sample["x"], sample["y"]

    spec = {
        "format": "neurolens-weights-v1",
        "name": model.name,
        "inputShape": list(model.input_shape[1:]),
        "layers": [layer_spec(l) for l in model.layers],
    }
    spec["totalParams"] = int(model.count_params())

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    OUT_MODEL.write_text(json.dumps(spec, separators=(",", ":")))

    # Per-layer outputs for the parity check.
    probe = keras.Model(model.inputs, [l.output for l in model.layers])
    xs = x[:PARITY_IMAGES]
    outs = probe.predict(xs, verbose=0)
    parity = {
        "count": PARITY_IMAGES,
        "labels": y[:PARITY_IMAGES].tolist(),
        "images": tensor(xs),
        "probabilities": tensor(outs[-1]),
        "predictions": outs[-1].argmax(axis=1).tolist(),
        # Every intermediate activation, for the first LAYER_IMAGES images only.
        "layerImages": LAYER_IMAGES,
        "layers": {l.name: tensor(o[:LAYER_IMAGES]) for l, o in zip(model.layers, outs)},
    }
    OUT_PARITY.parent.mkdir(parents=True, exist_ok=True)
    OUT_PARITY.write_text(json.dumps(parity, separators=(",", ":")))

    print(f"Model:  {spec['name']}  ({spec['totalParams']:,} params)")
    for l in spec["layers"]:
        shapes = ", ".join(f"{k}{tuple(v['shape'])}" for k, v in l["weights"].items()) or "-"
        print(f"  {l['name']:<8} {l['type']:<13} out={tuple(l['outputShape'])!s:<14} weights: {shapes}")
    print(f"Wrote {OUT_MODEL.relative_to(ROOT)}  ({OUT_MODEL.stat().st_size / 1024:.1f} KB)")
    print(f"Wrote {OUT_PARITY.relative_to(ROOT)}  ({OUT_PARITY.stat().st_size / 1024:.1f} KB)")
    print(f"Python predictions for parity images: {parity['predictions']}")
    print(f"True labels:                          {parity['labels']}")


if __name__ == "__main__":
    main()
