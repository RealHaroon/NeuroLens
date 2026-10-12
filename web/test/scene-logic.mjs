// Checks the data mapping behind the 3D scene (no WebGL needed):
//  - every layout panel maps one-to-one onto its layer's flat data
//  - conv receptive fields + bias reproduce the model's conv activations exactly
//  - pool windows, flatten order and dense top-k agree with the model
//
// Usage:  npm run test:scene

import { readFileSync } from 'node:fs';
import * as tf from '../src/tf.js';
import { buildModel, decodeArray } from '../src/model.js';
import { createInspector } from '../src/activations.js';
import { buildLayout } from '../src/viz/layout.js';
import { convInbound, denseTopK, poolInbound, unflatten } from '../src/viz/connections.js';

const read = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url), 'utf8'));
await tf.setBackend('cpu');
const inspector = createInspector(buildModel(read('../public/model/model.json')));
const parity = read('./fixtures/parity.json');
const acts = await inspector.run(decodeArray(parity.images).subarray(0, 784));
const W = inspector.weights;
let failed = false;
const check = (ok, msg) => {
  if (!ok) failed = true;
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${msg}`);
};

// 1. Layout panels cover each layer's data exactly once.
console.log('Layout:');
const L = buildLayout();
const sizes = {
  input: 784, conv1: 6272, pool1: 1568, conv2: 3136, pool2: 784,
  conv1Kernels: W.conv1.kernel.length, conv2Kernels: W.conv2.kernel.length,
};
for (const s of L.stations) {
  for (const [name, panels] of Object.entries(s.layers ?? {})) {
    const seen = new Uint8Array(sizes[name]);
    let dup = 0;
    for (const p of panels) {
      for (let r = 0; r < p.rows; r++) for (let c = 0; c < p.cols; c++) dup += seen[p.index(r, c)]++ ? 1 : 0;
    }
    check(dup === 0 && seen.every((v) => v === 1), `${name}: ${panels.length} panels cover all ${sizes[name]} values once`);
  }
}
const order = L.stations.map((s) => s.left);
check(order.every((x, i) => i === 0 || x > L.stations[i - 1].right), `${L.stations.length} stations left-to-right, no overlap (width ${L.width.toFixed(0)})`);

// 2. Conv receptive fields reproduce the activations.
console.log('Connections:');
function convError(layer, src, n) {
  const [h, w, C] = acts[layer].shape;
  let worst = 0;
  for (let t = 0; t < n; t++) {
    const r = (t * 7) % h, c = (t * 11) % w, co = t % C;
    const sum = convInbound(W[layer], acts[src], r, c, co).reduce((a, e) => a + e.contribution, 0) + W[layer].bias[co];
    worst = Math.max(worst, Math.abs(sum - acts[layer].data[(r * w + c) * C + co]));
  }
  return worst;
}
const e1 = convError('conv1', 'input', 400);
const e2 = convError('conv2', 'pool1', 400);
check(e1 < 1e-4, `conv1: Σ w·x + b over 3×3 window matches activation (max err ${e1.toExponential(1)})`);
check(e2 < 1e-4, `conv2: Σ w·x + b over 3×3×8 window matches activation (max err ${e2.toExponential(1)})`);

let poolOk = true;
for (const [pool, relu] of [['pool1', 'relu1'], ['pool2', 'relu2']]) {
  const [h, w, C] = acts[pool].shape;
  for (let i = 0; i < h * w * C; i += 13) {
    const ch = i % C, c = Math.floor(i / C) % w, r = Math.floor(i / C / w);
    const win = poolInbound(acts[relu], r, c, ch);
    const max = win.find((e) => e.isMax).value;
    if (Math.abs(max - acts[pool].data[i]) > 1e-6 || win.filter((e) => e.isMax).length !== 1) poolOk = false;
  }
}
check(poolOk, 'pool windows: the marked max equals the pooled value');

let flatOk = true;
for (let i = 0; i < 784; i++) {
  const { r, c, ch } = unflatten(i);
  if (acts.flatten.data[i] !== acts.pool2.data[(r * 7 + c) * 16 + ch]) flatOk = false;
}
check(flatOk, 'flatten[i] = pool2(r, c, ch) for all 784 cells');

let topOk = true;
let fullErr = 0;
for (let j = 0; j < 64; j++) {
  const top = denseTopK(W.dense1, acts.flatten.data, j, 8);
  const brute = Array.from({ length: 784 }, (_, i) => Math.abs(W.dense1.kernel[i * 64 + j] * acts.flatten.data[i])).sort((a, b) => b - a);
  top.forEach((e, n) => {
    if (Math.abs(Math.abs(e.contribution) - brute[n]) > 1e-7) topOk = false;
  });
  const all = denseTopK(W.dense1, acts.flatten.data, j, 784).reduce((a, e) => a + e.contribution, 0);
  fullErr = Math.max(fullErr, Math.abs(all + W.dense1.bias[j] - acts.dense1.data[j]));
}
check(topOk, 'dense top-8 matches a brute-force sort for all 64 neurons');
check(fullErr < 1e-3, `dense1: Σ of all w·x + b matches activation (max err ${fullErr.toExponential(1)})`);

console.log(failed ? '\nSCENE LOGIC FAILED' : '\nSCENE LOGIC OK');
process.exit(failed ? 1 : 0);
