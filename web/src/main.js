import * as tf from './tf.js';
import { loadModel } from './model.js';
import { createInspector, STAGES } from './activations.js';
import { canvasToGray, preprocess } from './preprocess.js';
import { createDrawpad } from './drawpad.js';
import { OVERVIEW, STEPS, paramsFor } from './ui/explain.js';

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
    renderAbout();
    setStatus('Draw a digit to begin');
    window.neurolens.last = null;
    return;
  }
  const result = await inspector.run(image);
  lastResult = result;
  renderProbs(result.softmax.data);
  renderLayerTable(result);
  scene3d?.update(result);
  renderAbout();
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
  renderAbout();
  await loadScene();
}

const STATION_NAMES = {
  input: 'Input', conv1: 'Conv1', pool1: 'Pool1', conv2: 'Conv2', pool2: 'Pool2',
  flatten: 'Flatten', dense1: 'Dense', output: 'Output',
};

async function loadScene() {
  const sceneEl = $('scene');
  try {
    const [{ createNetworkScene }, { createPlayer }] = await Promise.all([
      import('./viz/scene.js'),
      import('./ui/player.js'),
    ]);
    scene3d = createNetworkScene(sceneEl, inspector, { onFocus: setAbout });
    player = createPlayer(scene3d, { onChange: renderPlayer });
    $('scene-status').remove();
  } catch (err) {
    console.error(err);
    $('scene-status').textContent = `3D view unavailable: ${err.message}`;
    return;
  }
  window.neurolens.scene = scene3d;
  window.neurolens.player = player;
  if (lastResult) scene3d.update(lastResult);

  const topk = $('topk');
  const syncTopK = () => {
    $('topk-value').textContent = topk.value;
    scene3d.setTopK(Number(topk.value));
  };
  topk.addEventListener('input', syncTopK);
  syncTopK();

  $('play').addEventListener('click', () => player.toggle());
  $('step-back').addEventListener('click', () => player.step(-1));
  $('step-forward').addEventListener('click', () => player.step(1));
  $('speed').addEventListener('change', (e) => player.setSpeed(Number(e.target.value)));
  $('kernel-slide').addEventListener('change', (e) => player.setKernelSlide(e.target.checked));
  $('unroll').addEventListener('click', async () => {
    player.goto('flatten');
    await scene3d.focus('flatten');
    await scene3d.playFlatten();
  });
  $('reset-view').addEventListener('click', () => {
    player.stop();
    scene3d.overview();
  });
  $('stations').replaceChildren(
    ...scene3d.stations.map((id) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.dataset.station = id;
      b.textContent = STATION_NAMES[id];
      b.addEventListener('click', () => {
        player.goto(id);
        scene3d.focus(id);
      });
      return b;
    }),
  );

  // Keyboard: Space = play/pause, arrows = step (unless typing in a control).
  window.addEventListener('keydown', (e) => {
    if (e.target.closest('input, select, textarea') || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === ' ' && !e.target.closest('button')) {
      e.preventDefault();
      player.toggle();
    } else if (e.key === 'ArrowRight') {
      player.step(1);
    } else if (e.key === 'ArrowLeft') {
      player.step(-1);
    }
  });
  renderPlayer(player.state());
}

// --- playback + explanation panel ----------------------------------------------
let player = null;
let aboutStation = null; // null = whole network

function renderPlayer({ index, station, playing, atEnd }) {
  const btn = $('play');
  btn.classList.toggle('playing', playing);
  const label = playing ? 'Pause' : index >= 0 && !atEnd ? 'Resume' : atEnd ? 'Replay forward pass' : 'Play forward pass';
  $('play-label').textContent = label;
  btn.setAttribute('aria-label', label);
  $('step-back').disabled = index <= 0;
  $('step-forward').disabled = atEnd;
  $('play-status').textContent = index >= 0 ? `Layer ${index + 1} of ${scene3d.stations.length}: ${STATION_NAMES[station]}` : '';
  // Playback (including the kernel slide, which frames two layers) drives the panel.
  if (station && station !== aboutStation) setAbout(station);
  for (const b of $('stations').children) {
    if (b.dataset.station === station) b.setAttribute('aria-current', 'step');
    else b.removeAttribute('aria-current');
  }
}

function setAbout(station) {
  aboutStation = station;
  renderAbout();
}

function renderAbout() {
  const info = aboutStation ? STEPS[aboutStation] : OVERVIEW;
  const { total, parts } = paramsFor(info, inspector.spec);
  $('about-title').textContent = info.title;
  $('about-shape').textContent = info.shape;
  $('about-params').textContent = total.toLocaleString();
  const share = aboutStation && total ? ` (${((total / inspector.spec.totalParams) * 100).toFixed(1)}% of the model)` : '';
  $('about-body').innerHTML =
    info.body.map((p) => `<p>${p}</p>`).join('') +
    (parts.length ? `<p class="param-parts">${parts.join('<br>')}${share}</p>` : '<p class="param-parts">No learned parameters.</p>');
  $('about-live').innerHTML = lastResult ? info.live(lastResult) : 'Draw a digit to see live numbers for this layer.';
}

boot().catch((err) => {
  console.error(err);
  setStatus(`Failed to load model: ${err.message}`);
});
