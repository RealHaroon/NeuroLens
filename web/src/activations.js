// One forward pass -> every intermediate activation, plus the learned weights.
//
// A multi-output tf.Model shares the trained layers with the Sequential model,
// so a single predict() returns all 11 layer outputs. Results are copied into
// plain Float32Arrays and every tensor is disposed before returning.

import * as tf from './tf.js';
import { decodeArray } from './model.js';

/** Stage names in forward order; 'input' is the 28x28x1 image itself. */
export const STAGES = [
  'input', 'conv1', 'relu1', 'pool1', 'conv2', 'relu2', 'pool2',
  'flatten', 'dense1', 'relu3', 'dense2', 'softmax',
];

export function createInspector({ model, spec }) {
  const layerNames = spec.layers.map((l) => l.name);
  const probe = tf.model({
    inputs: model.inputs,
    outputs: layerNames.map((name) => model.getLayer(name).output),
    name: 'activation_probe',
  });

  const shapes = { input: spec.inputShape };
  for (const l of spec.layers) shapes[l.name] = l.outputShape;

  /**
   * @param {Float32Array} image 784 values in [0,1], row-major 28x28
   * @returns {Promise<{[stage: string]: {data: Float32Array, shape: number[]}}>}
   *   Data is the batch-1 activation, flattened in NHWC (row, col, channel) order.
   */
  async function run(image) {
    const outputs = tf.tidy(() => probe.predict(tf.tensor4d(image, [1, ...spec.inputShape])));
    try {
      const arrays = await Promise.all(outputs.map((t) => t.data()));
      const result = { input: { data: Float32Array.from(image), shape: shapes.input } };
      layerNames.forEach((name, i) => {
        result[name] = { data: arrays[i], shape: shapes[name] };
      });
      return result;
    } finally {
      tf.dispose(outputs);
    }
  }

  return { run, shapes, weights: extractWeights(spec), spec };
}

/**
 * Learned parameters as plain arrays, keyed by layer name:
 *   conv: kernel shape [kh, kw, cin, cout], dense: kernel shape [in, out].
 * Decoded straight from the exported spec, so no tensors are kept alive.
 */
export function extractWeights(spec) {
  const weights = {};
  for (const l of spec.layers) {
    if (!l.weights.kernel) continue;
    weights[l.name] = {
      kernel: decodeArray(l.weights.kernel),
      kernelShape: l.weights.kernel.shape,
      bias: l.weights.bias ? decodeArray(l.weights.bias) : null,
    };
  }
  return weights;
}

/** kernel[ky][kx][ci][co] for a conv layer's flattened [kh, kw, cin, cout] kernel. */
export function convWeight({ kernel, kernelShape }, ky, kx, ci, co) {
  const [, kw, cin, cout] = kernelShape;
  return kernel[((ky * kw + kx) * cin + ci) * cout + co];
}

/** kernel[i][j] for a dense layer's flattened [in, out] kernel. */
export function denseWeight({ kernel, kernelShape }, i, j) {
  return kernel[i * kernelShape[1] + j];
}

/** Value at (row, col, channel) of an [h, w, c] activation. */
export function at({ data, shape }, y, x, c) {
  return data[(y * shape[1] + x) * shape[2] + c];
}

/** Count of trainable parameters per layer (for the explanation panel). */
export function paramCount(layerSpec) {
  return Object.values(layerSpec.weights).reduce((n, w) => n + w.shape.reduce((a, b) => a * b, 1), 0);
}
