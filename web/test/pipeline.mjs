// Phase 2 checks, run in Node with the same modules the browser uses:
//  1. The multi-output activation probe returns every layer, matching Python.
//  2. Repeated forward passes leak no tensors.
//  3. MNIST-style preprocessing recovers digits that were upscaled, inverted,
//     shrunk, or moved off-center (as a user drawing/upload would be).
//
// Usage:  npm run test:pipeline

import { readFileSync } from 'node:fs';
import * as tf from '../src/tf.js';
import { buildModel, decodeArray } from '../src/model.js';
import { createInspector, STAGES } from '../src/activations.js';
import { preprocess } from '../src/preprocess.js';

const read = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url), 'utf8'));
const TOL = 1e-4;

await tf.setBackend('cpu');
const loaded = buildModel(read('../public/model/model.json'));
const inspector = createInspector(loaded);
const parity = read('./fixtures/parity.json');
const images = decodeArray(parity.images);
const imageAt = (i) => images.subarray(i * 784, (i + 1) * 784);
let failed = false;

// 1. Probe vs Python, first image.
const acts = await inspector.run(imageAt(0));
console.log('Activation probe (image 0):');
for (const stage of STAGES) {
  const { data, shape } = acts[stage];
  let line = `  ${stage.padEnd(8)} [${shape}]`.padEnd(28) + `min ${min(data).toFixed(3).padStart(7)}  max ${max(data).toFixed(3).padStart(6)}`;
  if (stage !== 'input') {
    const ref = decodeArray(parity.layers[stage]).subarray(0, data.length);
    const d = maxAbsDiff(data, ref);
    if (d > TOL) failed = true;
    line += `   vs Python max |Δ| ${d.toExponential(1)}`;
  }
  console.log(line);
}
const negBefore = acts.conv1.data.filter((v) => v < 0).length;
const negAfter = acts.relu1.data.filter((v) => v < 0).length;
console.log(`  conv1 has ${negBefore} negative values; relu1 has ${negAfter}`);
if (negAfter !== 0) failed = true;

const w = inspector.weights;
console.log(`Weights: ${Object.entries(w).map(([k, v]) => `${k}[${v.kernelShape}]`).join(' ')}`);

// 2. Leak check.
const before = tf.memory().numTensors;
for (let i = 0; i < 50; i++) await inspector.run(imageAt(i % parity.count));
const after = tf.memory().numTensors;
console.log(`\nTensor count before/after 50 runs: ${before} / ${after}`);
if (after !== before) failed = true;

// 3. Preprocessing robustness.
const variants = {
  'MNIST as-is': (img) => toGray(img, 28),
  'upscaled 280x280': (img) => upscale(img, 10),
  'inverted (ink on paper)': (img) => invertGray(upscale(img, 10)),
  'small, top-left corner': (img) => place(img, 280, 3, 5, 5),
  'large, bottom-right': (img) => place(img, 400, 12, 60, 40),
};
console.log('\nPreprocessing robustness (20 MNIST test digits):');
for (const [name, make] of Object.entries(variants)) {
  let correct = 0;
  let inverted = 0;
  for (let i = 0; i < parity.count; i++) {
    const res = preprocess(make(imageAt(i)));
    if (res.inverted) inverted++;
    const probs = (await inspector.run(res.image)).softmax.data;
    if (argmax(probs) === parity.labels[i]) correct++;
  }
  const ok = correct >= parity.count - 1;
  if (!ok) failed = true;
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${name.padEnd(26)} ${correct}/${parity.count} correct, inverted ${inverted}`);
}

const empty = preprocess(toGray(new Float32Array(784), 28));
console.log(`  ${empty.empty ? 'ok  ' : 'FAIL'} blank image reported as empty`);
if (!empty.empty) failed = true;

console.log(failed ? '\nPIPELINE FAILED' : '\nPIPELINE OK');
process.exit(failed ? 1 : 0);

// --- helpers -----------------------------------------------------------------
function toGray(img, size) {
  return { data: Float32Array.from(img), width: size, height: size };
}
function upscale(img, f) {
  return place(img, 28 * f, f, 0, 0);
}
/** Draw a 28x28 image scaled by `f` (nearest) at (ox, oy) on a black canvas of `size`. */
function place(img, size, f, ox, oy) {
  const data = new Float32Array(size * size);
  for (let y = 0; y < 28 * f; y++) {
    for (let x = 0; x < 28 * f; x++) {
      data[(y + oy) * size + (x + ox)] = img[Math.floor(y / f) * 28 + Math.floor(x / f)];
    }
  }
  return { data, width: size, height: size };
}
function invertGray(g) {
  return { ...g, data: g.data.map((v) => 1 - v) };
}
function argmax(a) {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
  return best;
}
function min(a) { return a.reduce((m, v) => Math.min(m, v), Infinity); }
function max(a) { return a.reduce((m, v) => Math.max(m, v), -Infinity); }
function maxAbsDiff(a, b) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}
