// Rebuilds the trained Keras CNN in TF.js from the JSON written by
// training/export_weights.py, then loads the exported weights into it.

import * as tf from '@tensorflow/tfjs';

export const MODEL_FORMAT = 'neurolens-weights-v1';

/** Decode a {shape, dtype, data: base64 little-endian float32} record. */
export function decodeArray({ data }) {
  const bin = atob(data);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

export function decodeTensor(record) {
  return tf.tensor(decodeArray(record), record.shape, 'float32');
}

function createLayer(spec, isFirst, inputShape) {
  const base = { name: spec.name, ...(isFirst ? { inputShape } : {}) };
  switch (spec.type) {
    case 'conv2d':
      return tf.layers.conv2d({
        ...base,
        filters: spec.filters,
        kernelSize: spec.kernelSize,
        strides: spec.strides,
        padding: spec.padding,
        useBias: spec.useBias,
      });
    case 'relu':
      return tf.layers.reLU(base);
    case 'maxPooling2d':
      return tf.layers.maxPooling2d({
        ...base,
        poolSize: spec.poolSize,
        strides: spec.strides,
        padding: spec.padding,
      });
    case 'flatten':
      return tf.layers.flatten(base);
    case 'dense':
      return tf.layers.dense({ ...base, units: spec.units, useBias: spec.useBias });
    case 'softmax':
      return tf.layers.softmax(base);
    default:
      throw new Error(`Unsupported layer type "${spec.type}" (${spec.name})`);
  }
}

/**
 * Build a tf.Sequential from an exported spec and load its weights.
 * Returns { model, spec }. Keras and TF.js share kernel layouts
 * (conv: [kh, kw, cin, cout], dense: [in, out]), so weights load as-is.
 */
export function buildModel(spec) {
  if (spec.format !== MODEL_FORMAT) {
    throw new Error(`Unexpected model format "${spec.format}"`);
  }
  const model = tf.sequential({ name: spec.name });
  spec.layers.forEach((layerSpec, i) => {
    model.add(createLayer(layerSpec, i === 0, spec.inputShape));
  });

  spec.layers.forEach((layerSpec) => {
    const { kernel, bias } = layerSpec.weights;
    if (!kernel) return;
    const weights = [decodeTensor(kernel), ...(bias ? [decodeTensor(bias)] : [])];
    model.getLayer(layerSpec.name).setWeights(weights);
    tf.dispose(weights);
  });

  return { model, spec };
}

export async function loadModel(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load model from ${url}: ${res.status}`);
  return buildModel(await res.json());
}
