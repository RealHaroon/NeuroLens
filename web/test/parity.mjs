// Parity check: runs the MNIST test images exported by training/export_weights.py
// through the TF.js rebuild (src/model.js, the same code the browser uses) and
// compares predictions and every intermediate activation against Python's.
//
// Usage:  npm run test:parity

import { readFileSync } from 'node:fs';
import * as tf from '@tensorflow/tfjs';
import { buildModel, decodeArray, decodeTensor } from '../src/model.js';

const ACTIVATION_TOL = 1e-4;
const read = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url), 'utf8'));

await tf.setBackend('cpu');
const { model, spec } = buildModel(read('../public/model/model.json'));
const parity = read('./fixtures/parity.json');

let failed = false;
const images = decodeTensor(parity.images);

// 1. Predictions + probabilities for all images.
const probs = model.predict(images);
const jsPred = Array.from(probs.argMax(1).dataSync());
const probDiff = maxAbsDiff(probs.dataSync(), decodeArray(parity.probabilities));
const mismatches = jsPred.filter((p, i) => p !== parity.predictions[i]).length;

console.log(`TF.js backend: ${tf.getBackend()}, ${model.countParams().toLocaleString()} params`);
console.log(`Python predictions: ${parity.predictions.join(' ')}`);
console.log(`TF.js predictions:  ${jsPred.join(' ')}`);
console.log(`True labels:        ${parity.labels.join(' ')}`);
console.log(`Prediction mismatches: ${mismatches}/${parity.count}, max |Δprob| = ${probDiff.toExponential(2)}\n`);
if (mismatches > 0 || probDiff > ACTIVATION_TOL) failed = true;

// 2. Every intermediate activation, layer by layer, for the first few images.
let x = images.slice(0, parity.layerImages);
for (const { name } of spec.layers) {
  const next = model.getLayer(name).apply(x);
  x.dispose();
  x = next;
  const diff = maxAbsDiff(x.dataSync(), decodeArray(parity.layers[name]));
  const ok = diff <= ACTIVATION_TOL;
  if (!ok) failed = true;
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${name.padEnd(8)} shape=[${x.shape.slice(1)}]`.padEnd(40) +
    `max |Δ| = ${diff.toExponential(2)}`);
}
tf.dispose([x, images, probs]);

console.log(failed ? '\nPARITY FAILED' : '\nPARITY OK: TF.js matches Python at every layer');
process.exit(failed ? 1 : 0);

function maxAbsDiff(a, b) {
  if (a.length !== b.length) throw new Error(`length mismatch ${a.length} vs ${b.length}`);
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}
