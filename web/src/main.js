import * as tf from './tf.js';
import { loadModel } from './model.js';
import { createInspector, STAGES } from './activations.js';
import { canvasToGray, preprocess } from './preprocess.js';
import { createDrawpad } from './drawpad.js';

const MAX_UPLOAD_DIM = 512;

const $ = (id) => document.getElementById(id);
const statusEl = $('status');
const previewCtx = $('preview').getContext('2d');

let inspector = null;
let scene3d = null; // loaded lazily so the input panel works before three.js arrives
let lastResult = null;
let source = null; // canvas the next inference reads from (drawpad or uploaded image)

const drawpad = createDrawpad($('drawpad'), {
  onStart: () => {
    if (source !== drawpad.canvas) {
      drawpad.clear(); // drawing after an upload starts fresh
      source = drawpad.canvas;
    }
  },
  onChange: () => schedule(),
});
source = drawpad.canvas;

$('clear').addEventListener('click', () => {
  drawpad.clear();
  source = drawpad.canvas;
  schedule();
});

$('upload').addEventListener('change', async (e) => {
  const file = e.target.files?.[0];
  e.target.value = '';
  if (!file) return;
  try {
    const img = await createImageBitmap(file);
    const s = Math.min(1, MAX_UPLOAD_DIM / Math.max(img.width, img.height));
    const off = document.createElement('canvas');
    off.width = Math.max(1, Math.round(img.width * s));
    off.height = Math.max(1, Math.round(img.height * s));
    off.getContext('2d', { willReadFrequently: true }).drawImage(img, 0, 0, off.width, off.height);
    img.close();
    drawpad.showImage(off);
    source = off;
    schedule();
  } catch (err) {
    setStatus(`Could not read that image (${err.message})`);
  }
});

// --- inference scheduling: at most one pass per frame, never overlapping -----
let scheduled = false;
let running = false;
let dirty = false;

function schedule() {
  if (!inspector || scheduled) return;
  scheduled = true;
  requestAnimationFrame(async () => {
    scheduled = false;
    if (running) {
      dirty = true;
      return;
    }
    running = true;
    try {
      await infer();
    } finally {
      running = false;
    }
    if (dirty) {
      dirty = false;
      schedule();
    }
  });
}

async function infer() {
  const t0 = performance.now();
  const { image, empty } = preprocess(canvasToGray(source));
  drawPreview(image);
  if (empty) {
    renderProbs(null);
    renderLayerTable(null);
    scene3d?.update(null);
    lastResult = null;
    setStatus('Draw a digit to begin');
    window.neurolens.last = null;
    return;
  }
  const result = await inspector.run(image);
  lastResult = result;
  renderProbs(result.softmax.data);
  renderLayerTable(result);
  scene3d?.update(result);
  const ms = performance.now() - t0;
  setStatus(`${tf.getBackend()} · ${ms.toFixed(1)} ms · ${tf.memory().numTensors} tensors`);
  window.neurolens.last = result;
  window.dispatchEvent(new CustomEvent('neurolens:result', { detail: result }));
}

// --- rendering ---------------------------------------------------------------
function drawPreview(image) {
  const px = previewCtx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const v = Math.round(image[i] * 255);
    px.data.set([v, v, v, 255], i * 4);
  }
  previewCtx.putImageData(px, 0, 0);
}

const probRows = Array.from({ length: 10 }, (_, d) => {
  const li = document.createElement('li');
  li.innerHTML = `<span class="digit">${d}</span><span class="track"><span class="fill"></span></span><span class="pct">–</span>`;
  $('probs').append(li);
  return { li, fill: li.querySelector('.fill'), pct: li.querySelector('.pct') };
});

function renderProbs(probs) {
  let top = -1;
  if (probs) top = probs.indexOf(Math.max(...probs));
  probRows.forEach((row, d) => {
    const p = probs ? probs[d] : 0;
    row.fill.style.width = `${(p * 100).toFixed(1)}%`;
    row.pct.textContent = probs ? `${(p * 100).toFixed(1)}%` : '–';
    row.li.classList.toggle('top', d === top);
  });
}

function renderLayerTable(result) {
  const body = $('layers').tBodies[0];
  body.replaceChildren(
    ...STAGES.map((stage) => {
      const tr = document.createElement('tr');
      const shape = inspector.shapes[stage].join('×');
      if (!result) {
        tr.innerHTML = `<td>${stage}</td><td>${shape}</td><td>–</td><td>–</td><td>–</td><td>–</td>`;
        return tr;
      }
      const d = result[stage].data;
      let mn = Infinity, mx = -Infinity, sum = 0, zeros = 0;
      for (const v of d) {
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v;
        if (v === 0) zeros++;
      }
      tr.innerHTML = `<td>${stage}</td><td>${shape}</td><td>${mn.toFixed(3)}</td><td>${mx.toFixed(3)}</td>` +
        `<td>${(sum / d.length).toFixed(3)}</td><td>${((zeros / d.length) * 100).toFixed(0)}%</td>`;
      return tr;
    }),
  );
}

function setStatus(text) {
  statusEl.textContent = text;
}

// --- boot ----------------------------------------------------------------------
// Debug/test hook: the latest activations and the inspector (weights, shapes).
window.neurolens = { last: null, inspector: null, ready: false };

async function boot() {
  // The model is tiny: the CPU backend runs a full pass in a few ms, reads back
  // activations for free, and leaves the GPU to Three.js.
  await tf.setBackend('cpu');
  const loaded = await loadModel(`${import.meta.env.BASE_URL}model/model.json`);
  inspector = createInspector(loaded);
  window.neurolens.inspector = inspector;
  window.neurolens.ready = true;
  renderProbs(null);
  renderLayerTable(null);
  setStatus('Draw a digit to begin');
  await loadScene();
}

const STATION_NAMES = {
  input: 'Input', conv1: 'Conv1', pool1: 'Pool1', conv2: 'Conv2', pool2: 'Pool2',
  flatten: 'Flatten', dense1: 'Dense', output: 'Output',
};

async function loadScene() {
  const sceneEl = $('scene');
  try {
    const { createNetworkScene } = await import('./viz/scene.js');
    scene3d = createNetworkScene(sceneEl, inspector);
    $('scene-status').remove();
  } catch (err) {
    console.error(err);
    $('scene-status').textContent = `3D view unavailable: ${err.message}`;
    return;
  }
  window.neurolens.scene = scene3d;
  if (lastResult) scene3d.update(lastResult);

  const topk = $('topk');
  const syncTopK = () => {
    $('topk-value').textContent = topk.value;
    scene3d.setTopK(Number(topk.value));
  };
  topk.addEventListener('input', syncTopK);
  syncTopK();
  $('unroll').addEventListener('click', async () => {
    await scene3d.focus('flatten');
    await scene3d.playFlatten();
  });
  $('reset-view').addEventListener('click', () => scene3d.overview());
  $('stations').replaceChildren(
    ...scene3d.stations.map((id) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = STATION_NAMES[id];
      b.addEventListener('click', () => scene3d.focus(id));
      return b;
    }),
  );
}

boot().catch((err) => {
  console.error(err);
  setStatus(`Failed to load model: ${err.message}`);
});
