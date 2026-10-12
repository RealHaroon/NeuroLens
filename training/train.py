"""Train the small NeuroLens CNN on MNIST and save it as a Keras model.

Every activation is its own layer (not fused into Conv2D/Dense) so the web app
can read both pre- and post-ReLU values.

Usage:  python train.py [--epochs 5]
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
from keras import layers

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
MODEL_PATH = ARTIFACTS / "mnist_cnn.keras"
SEED = 42


def build_model() -> keras.Model:
    return keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1), name="input"),
            layers.Conv2D(8, 3, padding="same", name="conv1"),
            layers.ReLU(name="relu1"),
            layers.MaxPooling2D(2, name="pool1"),
            layers.Conv2D(16, 3, padding="same", name="conv2"),
            layers.ReLU(name="relu2"),
            layers.MaxPooling2D(2, name="pool2"),
            layers.Flatten(name="flatten"),
            layers.Dense(64, name="dense1"),
            layers.ReLU(name="relu3"),
            layers.Dense(10, name="dense2"),
            layers.Softmax(name="softmax"),
        ],
        name="neurolens_cnn",
    )


def load_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test = (x_test.astype("float32") / 255.0)[..., None]
    return (x_train, y_train), (x_test, y_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    keras.utils.set_random_seed(SEED)
    (x_train, y_train), (x_test, y_test) = load_mnist()

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        verbose=2,
    )

    loss, acc = model.evaluate(x_test, y_test, batch_size=512, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}  (loss {loss:.4f})")
    if acc < 0.98:
        print("WARNING: test accuracy is below the 98% target")

    ARTIFACTS.mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH.relative_to(HERE.parent)}")

    # Keep a tiny test-set sample for the export/parity steps.
    np.savez_compressed(ARTIFACTS / "test_sample.npz", x=x_test[:200], y=y_test[:200])


if __name__ == "__main__":
    main()
