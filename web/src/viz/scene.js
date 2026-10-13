// The 3D network: builds every station once, then each new forward pass only
// rewrites instance colors/matrices and the top-k connection buffers.
//
// Rendering is on demand: a frame is drawn only when the camera moves, data
// changes, or an animation is running, which keeps an idle laptop cool.

import {
  AmbientLight, Box3, BoxGeometry, Color, DirectionalLight, EdgesGeometry, InstancedBufferAttribute, InstancedMesh,
  LineBasicMaterial, LineSegments, MeshBasicMaterial, MOUSE, PerspectiveCamera, Plane, Raycaster, Scene, TOUCH, Vector2,
  Vector3, WebGLRenderer,
} from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DObject, CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';
import { buildLayout } from './layout.js';
import { BOX, GridLayer, instanced, LineBatch, NeuronLayer, writeMatrix } from './primitives.js';
import { ACCENT, cssGradient, LUTS, lutIndex, MUTED, SCENE_BG, SIGN_COLORS } from './colors.js';
import { convInbound, denseTopK, poolInbound, unflatten } from './connections.js';
import { createSteps } from './steps.js';

const MAX_K = 32;
const FLATTEN_SECONDS = 2.6;
const FLY_SECONDS = 0.9;
const DIM = 0.07; // brightness of stations the forward pass hasn't reached yet

const STATION_LABELS = {
  input: ['Input', '28×28×1'],
  conv1: ['Conv1', '8 filters 3×3 → 28×28×8'],
  pool1: ['ReLU + Pool', '14×14×8'],
  conv2: ['Conv2', '16 filters 3×3×8 → 14×14×16'],
  pool2: ['ReLU + Pool', '7×7×16'],
  flatten: ['Flatten', '784'],
  dense1: ['Dense', '64 + ReLU'],
  output: ['Output', '10 · softmax'],
};

/**
 * @param {HTMLElement} container
 * @param {object} inspector from createInspector()
 * @param {{onFocus?: (stationId: string | null) => void}} [options] onFocus fires when the
 *   camera is sent to a station (null = overview), e.g. to update an explanation panel.
 */
export function createNetworkScene(container, inspector, { onFocus } = {}) {
  const L = buildLayout();
  const W = inspector.weights;

  // --- renderer, camera, controls ---------------------------------------------
  const renderer = new WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
  renderer.domElement.className = 'scene-canvas';
  container.append(renderer.domElement);

  const labelRenderer = new CSS2DRenderer();
  labelRenderer.domElement.className = 'scene-labels';
  container.append(labelRenderer.domElement);

  const scene = new Scene();
  scene.background = new Color(SCENE_BG);
  // three.js lights are physically based (Lambert divides by π), so intensities
  // are scaled by π: front faces then show their colormap color at ~full value.
  scene.add(new AmbientLight(0xffffff, 0.35 * Math.PI));
  const sun = new DirectionalLight(0xffffff, 0.75 * Math.PI);
  sun.position.set(-0.3, 0.5, 1);
  scene.add(sun);

  const camera = new PerspectiveCamera(40, 1, 0.5, 5000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.screenSpacePanning = true;
  controls.minDistance = 8;
  controls.maxDistance = 1600;
  // The network reads left to right, so dragging slides the view (pan) like a map;
  // rotating is the deliberate action (right-drag, Shift/Ctrl+drag, or "Rotate" mode).
  // Wheel and pinch zoom are handled below, not by OrbitControls.
  controls.enableZoom = true; // still needed for two-finger touch pinch
  setDragMode('move');

  function setDragMode(mode) {
    const rotate = mode === 'rotate';
    controls.mouseButtons = { LEFT: rotate ? MOUSE.ROTATE : MOUSE.PAN, MIDDLE: MOUSE.DOLLY, RIGHT: rotate ? MOUSE.PAN : MOUSE.ROTATE };
    controls.touches = { ONE: rotate ? TOUCH.ROTATE : TOUCH.PAN, TWO: TOUCH.DOLLY_PAN };
  }

  let needsRender = true;
  const invalidate = () => {
    needsRender = true;
  };
  controls.addEventListener('change', invalidate);

  // --- layers --------------------------------------------------------------------
  const S = L.byId;
  const grids = {
    input: new GridLayer('input', S.input.layers.input, { colormap: 'grayscale', depth: [0.12, 1.6] }),
    conv1Kernels: new GridLayer('conv1Kernels', S.conv1.layers.conv1Kernels, { colormap: 'kernel', depth: [0.15, 0.9] }),
    conv1: new GridLayer('conv1', S.conv1.layers.conv1, { colormap: 'diverging' }),
    pool1: new GridLayer('pool1', S.pool1.layers.pool1, { colormap: 'sequential' }),
    conv2Kernels: new GridLayer('conv2Kernels', S.conv2.layers.conv2Kernels, { colormap: 'kernel', depth: [0.08, 0.45] }),
    conv2: new GridLayer('conv2', S.conv2.layers.conv2, { colormap: 'diverging' }),
    pool2: new GridLayer('pool2', S.pool2.layers.pool2, { colormap: 'sequential' }),
  };
  for (const g of Object.values(grids)) scene.add(g.mesh);
  grids.conv1Kernels.update(W.conv1.kernel, maxAbs(W.conv1.kernel));
  grids.conv2Kernels.update(W.conv2.kernel, maxAbs(W.conv2.kernel));

  const flatten = new FlattenColumn(S.flatten);
  scene.add(flatten.mesh);

  const dense = new NeuronLayer('dense1', S.dense1.neurons, S.dense1.radius, { colorFloor: 70 });
  const output = new NeuronLayer('output', S.output.neurons, S.output.radius, { colorFloor: 70 });
  scene.add(dense.mesh, output.mesh);

  const bars = instanced(BOX, 10);
  scene.add(bars);

  const links1 = new LineBatch(64 * MAX_K); // flatten -> dense1
  const links2 = new LineBatch(10 * MAX_K); // dense1 -> output
  const focusLinks = new LineBatch(1024); // hovered unit's inputs
  scene.add(links1.lines, links2.lines, focusLinks.lines);

  const outline = new LineSegments(new EdgesGeometry(new BoxGeometry(1, 1, 1)), new LineBasicMaterial({ color: 0xffffff }));
  outline.visible = false;
  scene.add(outline);

  // Step animations: window outlines (kernel / pooling windows) and flying max cells.
  const edgeBox = new EdgesGeometry(new BoxGeometry(1, 1, 1));
  const boxMaterial = new LineBasicMaterial({ color: 0xfab219 });
  const boxes = Array.from({ length: 20 }, () => {
    const b = new LineSegments(edgeBox, boxMaterial);
    b.visible = false;
    scene.add(b);
    return b;
  });
  const flyers = new InstancedMesh(BOX, new MeshBasicMaterial({ color: 0xffffff }), 16);
  flyers.instanceColor = new InstancedBufferAttribute(new Float32Array(16 * 3), 3);
  flyers.count = 0;
  flyers.frustumCulled = false;
  scene.add(flyers);
  const flyerColor = new Color(0xfab219);

  function showBox(i, center, size) {
    boxes[i].position.set(...center);
    boxes[i].scale.set(...size);
    boxes[i].visible = true;
  }
  function hideBoxes() {
    for (const b of boxes) b.visible = false;
  }
  function setFlyer(i, center, size, brightness) {
    writeMatrix(flyers.instanceMatrix.array, i, center[0], center[1], center[2], size, size, size);
    flyers.instanceColor.array.set([flyerColor.r * brightness, flyerColor.g * brightness, flyerColor.b * brightness], i * 3);
    flyers.instanceMatrix.needsUpdate = true;
    flyers.instanceColor.needsUpdate = true;
  }

  // --- labels --------------------------------------------------------------------
  const stationLabels = {};
  for (const s of L.stations) {
    const [title, sub] = STATION_LABELS[s.id];
    const el = labelEl('scene-label station', `<strong>${title}</strong><span>${sub}</span>`);
    const obj = new CSS2DObject(el);
    obj.position.set(s.center[0], s.height / 2 + 7, 0);
    scene.add(obj);
    stationLabels[s.id] = el;
  }
  const digitLabels = S.output.neurons.map(([x, y], d) => {
    const obj = new CSS2DObject(labelEl('scene-label digit', String(d)));
    obj.position.set(x + S.output.radius[1] + 2.2, y, 0);
    scene.add(obj);
    return obj;
  });
  const probLabels = S.output.neurons.map(([, y]) => {
    const obj = new CSS2DObject(labelEl('scene-label prob', ''));
    obj.position.set(S.output.bars.x, y, 0);
    scene.add(obj);
    return obj;
  });

  container.append(buildLegend());
  const tooltip = document.createElement('div');
  tooltip.className = 'scene-tooltip';
  tooltip.hidden = true;
  container.append(tooltip);
  const caption = document.createElement('div');
  caption.className = 'scene-caption';
  caption.hidden = true;
  container.append(caption);

  // --- station brightness (the forward pass "lights up" layer by layer) ----------
  const parts = {
    input: [grids.input.mesh],
    conv1: [grids.conv1Kernels.mesh, grids.conv1.mesh],
    pool1: [grids.pool1.mesh],
    conv2: [grids.conv2Kernels.mesh, grids.conv2.mesh],
    pool2: [grids.pool2.mesh],
    flatten: [flatten.mesh],
    dense1: [dense.mesh],
    output: [output.mesh, bars],
  };
  const levels = Object.fromEntries(L.stations.map((s) => [s.id, 1]));
  const targets = { ...levels };
  const linkDim = { links1: 1, links2: 1 }; // dimmed while a dense unit is focused
  let current = null;

  /** Light stations up to and including `index` (in forward order); null lights everything. */
  function setLit(index) {
    current = index === null ? null : L.stations[index]?.id ?? null;
    L.stations.forEach((s, i) => {
      targets[s.id] = index === null || i <= index ? 1 : DIM;
      stationLabels[s.id].classList.toggle('current', s.id === current);
    });
    invalidate();
  }

  function applyLevels() {
    for (const [id, meshes] of Object.entries(parts)) {
      for (const m of meshes) m.material.color.setScalar(levels[id]);
      stationLabels[id].style.opacity = 0.25 + 0.75 * levels[id];
    }
    links1.material.opacity = Math.min(levels.flatten, levels.dense1) * linkDim.links1;
    links2.material.opacity = Math.min(levels.dense1, levels.output) * linkDim.links2;
    for (const l of [...digitLabels, ...probLabels]) l.element.style.opacity = 0.25 + 0.75 * levels.output;
  }

  // --- data updates --------------------------------------------------------------
  let result = null;
  let k = 8;

  // Display overrides used by the step animations (viz/steps.js). They are
  // re-applied on every redraw, so live drawing during an animation stays correct.
  //   mods[grid] = { reveal(x, y) -> 0..1, reluT: 0..1, mask: { n, C, channels? } }
  const mods = {};
  const neuronMods = { dense1: null, output: null }; // (i) => radius/brightness scale
  let probOverride = null; // probabilities shown instead of the real softmax (race)
  const scratch = {};
  const kernelMax = { conv1Kernels: maxAbs(W.conv1.kernel), conv2Kernels: maxAbs(W.conv2.kernel) };

  function refreshGrid(name) {
    let data;
    let max;
    if (name === 'conv1Kernels' || name === 'conv2Kernels') {
      data = W[name.slice(0, 5)].kernel;
      max = kernelMax[name];
    } else {
      data = result?.[name].data ?? null;
      max = name === 'input' ? 1 : data ? maxAbs(data) : 0;
    }
    const m = mods[name];
    if (data && m && (m.reluT || m.mask)) {
      const buf = (scratch[name] ??= new Float32Array(data.length));
      buf.set(data);
      if (m.reluT) {
        for (let i = 0; i < buf.length; i++) if (buf[i] < 0) buf[i] *= 1 - m.reluT;
      }
      if (m.mask) {
        // Hide cells not yet "computed": cell index >= n, or visible(cell) false.
        const { n = 0, visible, C, channels } = m.mask;
        for (let i = visible ? 0 : n * C; i < buf.length; i++) {
          if (channels && !channels.has(i % C)) continue;
          if (!visible || !visible(Math.floor(i / C))) buf[i] = 0;
        }
      }
      data = buf;
    }
    grids[name].update(data, max, m?.reveal);
    invalidate();
  }

  function refreshNeurons() {
    dense.update(result?.relu3.data ?? null, result ? maxAbs(result.relu3.data) : 0, neuronMods.dense1);
    output.update(probOverride ?? result?.softmax.data ?? null, 1, neuronMods.output);
    updateBars();
    invalidate();
  }

  function refreshAll() {
    for (const name of Object.keys(grids)) refreshGrid(name);
    refreshNeurons();
  }

  function update(next) {
    result = next;
    if (!result) cancelAnimations();
    refreshAll();
    flatten.update(result?.flatten.data ?? null, result ? maxAbs(result.flatten.data) : 0);
    updateLinks();
    refreshFocus();
    invalidate();
  }

  function updateBars() {
    const probs = probOverride ?? result?.softmax.data;
    const top = probs && Math.max(...probs) > 0 ? argmax(probs) : -1;
    const { x, maxLength, height } = S.output.bars;
    const mat = bars.instanceMatrix.array;
    const col = bars.instanceColor.array;
    for (let d = 0; d < 10; d++) {
      const p = probs ? probs[d] : 0;
      const len = Math.max(0.15, p * maxLength);
      const y = S.output.neurons[d][1];
      writeMatrix(mat, d, x + len / 2, y, 0.6, len, height, 1.2);
      const c = d === top ? ACCENT : MUTED;
      col.set([c.r, c.g, c.b], d * 3);
      const el = probLabels[d].element;
      // Near-zero classes keep their bar but drop the text to avoid clutter.
      el.textContent = probs && p >= 0.001 ? `${(p * 100).toFixed(1)}%` : '';
      el.classList.toggle('top', d === top);
      digitLabels[d].element.classList.toggle('top', d === top);
      probLabels[d].position.x = x + len + 4;
    }
    bars.instanceMatrix.needsUpdate = true;
    bars.instanceColor.needsUpdate = true;
  }

  /** Top-k strongest incoming connections (|weight × input|) for every dense unit. */
  function updateLinks() {
    links1.clear();
    links2.clear();
    if (result) {
      const sets1 = Array.from({ length: 64 }, (_, j) => denseTopK(W.dense1, result.flatten.data, j, k));
      const sets2 = Array.from({ length: 10 }, (_, j) => denseTopK(W.dense2, result.relu3.data, j, k));
      const max1 = maxContribution(sets1);
      const max2 = maxContribution(sets2);
      sets1.forEach((set, j) => {
        const to = anchor(dense.cell(j), 'left');
        for (const e of set) pushLink(links1, flattenAnchor(e.i), to, e.contribution, max1);
      });
      sets2.forEach((set, j) => {
        const to = anchor(output.cell(j), 'left');
        for (const e of set) pushLink(links2, anchor(dense.cell(e.i), 'right'), to, e.contribution, max2);
      });
    }
    links1.commit();
    links2.commit();
  }

  function pushLink(batch, from, to, contribution, max, floor = 0.08) {
    const color = contribution >= 0 ? SIGN_COLORS.positive : SIGN_COLORS.negative;
    batch.push(from, to, color, floor + (1 - floor) * (max > 0 ? Math.abs(contribution) / max : 0));
  }

  function flattenAnchor(i) {
    return anchor(flatten.cell(i), 'right');
  }

  // --- hover / selection -------------------------------------------------------------
  const raycaster = new Raycaster();
  const ndc = new Vector2();
  const plane = new Plane(new Vector3(0, 0, 1), 0);
  const hitPoint = new Vector3();
  let hovered = null; // key of the unit under the pointer
  let pinned = null; // key of the clicked unit (wins over hover)

  function pickXY(x, y) {
    if (x >= S.output.left - 4) {
      const hit = output.pick(x, y);
      if (hit) return hit;
      const { x: bx, maxLength, height } = S.output.bars;
      if (x >= bx && x <= bx + maxLength + 6) {
        const d = S.output.neurons.findIndex(([, ny]) => Math.abs(ny - y) <= height / 2 + 0.5);
        if (d >= 0) return output.cell(d);
      }
      return null;
    }
    if (x >= S.dense1.left - 2) return dense.pick(x, y);
    if (x >= S.flatten.left - 2) return flatten.pick(x, y);
    for (const g of Object.values(grids)) {
      const hit = g.pick(x, y);
      if (hit) return hit;
    }
    return null;
  }

  function pickAt(clientX, clientY) {
    const rect = renderer.domElement.getBoundingClientRect();
    ndc.set(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
    raycaster.setFromCamera(ndc, camera);
    plane.constant = 0;
    if (!raycaster.ray.intersectPlane(plane, hitPoint)) return null;
    let hit = pickXY(hitPoint.x, hitPoint.y);
    // Cells extrude toward the camera: re-pick on the hit cell's front face to
    // correct the parallax of a tilted view.
    if (hit && hit.panel) {
      plane.constant = -(hit.center[2] + hit.size[2] / 2);
      if (raycaster.ray.intersectPlane(plane, hitPoint)) hit = pickXY(hitPoint.x, hitPoint.y) ?? hit;
    }
    return hit;
  }

  const keyOf = (hit) => (hit ? { layer: hit.layer, panelIndex: hit.panelIndex, r: hit.r, c: hit.c, index: hit.index } : null);
  const sameKey = (a, b) => a && b && a.layer === b.layer && a.panelIndex === b.panelIndex && a.r === b.r && a.c === b.c && a.index === b.index;

  function resolve(key) {
    if (!key) return null;
    if (grids[key.layer]) return grids[key.layer].cell(key.panelIndex, key.r, key.c);
    if (key.layer === 'flatten') return flatten.cell(key.index);
    if (key.layer === 'dense1') return dense.cell(key.index);
    if (key.layer === 'output') return output.cell(key.index);
    return null;
  }

  let pointer = { x: 0, y: 0 };
  let downAt = null;
  const canvas = renderer.domElement;
  canvas.addEventListener('pointermove', (e) => {
    pointer = { x: e.clientX, y: e.clientY };
    if (e.buttons || steps.phase()) return; // orbiting, or a step animation owns the highlight
    const key = keyOf(pickAt(e.clientX, e.clientY));
    if (!sameKey(key, hovered)) {
      hovered = key;
      refreshFocus();
    } else if (!pinned) {
      placeTooltip();
    }
  });
  canvas.addEventListener('pointerleave', () => {
    hovered = null;
    refreshFocus();
  });
  canvas.addEventListener('pointerdown', (e) => {
    downAt = { x: e.clientX, y: e.clientY };
  });
  canvas.addEventListener('pointerup', (e) => {
    if (!downAt || Math.hypot(e.clientX - downAt.x, e.clientY - downAt.y) > 4) return;
    const key = keyOf(pickAt(e.clientX, e.clientY));
    pinned = key && !sameKey(key, pinned) ? key : null;
    pointer = { x: e.clientX, y: e.clientY };
    refreshFocus();
  });
  canvas.addEventListener('dblclick', (e) => {
    const hit = pickAt(e.clientX, e.clientY);
    const station = hit ? stationOf(hit.layer) : null;
    if (station) focus(station);
  });
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && pinned) {
      pinned = null;
      refreshFocus();
    }
  });

  /** Redraw the outline, incoming-connection lines and tooltip for the focused unit. */
  function refreshFocus() {
    const hit = resolve(pinned ?? hovered);
    focusLinks.clear();
    linkDim.links1 = 1;
    linkDim.links2 = 1;
    if (!hit || !result) {
      outline.visible = false;
      tooltip.hidden = true;
      focusLinks.commit();
      invalidate();
      return;
    }
    outline.visible = true;
    outline.position.set(...hit.center);
    outline.scale.set(hit.size[0] * 1.08, hit.size[1] * 1.08, hit.size[2] * 1.08);

    const info = explain(hit);
    focusLinks.commit();
    tooltip.innerHTML = info;
    tooltip.hidden = false;
    tooltip.classList.toggle('pinned', !!pinned);
    placeTooltip();
    invalidate();
  }

  function placeTooltip() {
    if (tooltip.hidden) return;
    const rect = container.getBoundingClientRect();
    const tw = tooltip.offsetWidth;
    const th = tooltip.offsetHeight;
    let x = pointer.x - rect.left + 16;
    let y = pointer.y - rect.top + 16;
    if (x + tw > rect.width - 8) x = pointer.x - rect.left - tw - 16;
    if (y + th > rect.height - 8) y = rect.height - th - 8;
    tooltip.style.transform = `translate(${Math.max(8, x)}px, ${Math.max(8, y)}px)`;
  }

  /** Builds the tooltip HTML and fills focusLinks with the unit's incoming connections. */
  function explain(hit) {
    const R = result;
    const fmt = (v) => (Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(3));
    const row = (k, v) => `<div class="tt-row"><span>${k}</span><b>${v}</b></div>`;
    const card = (title, rows, note) => `<div class="tt-title">${title}</div>${rows.join('')}${note ? `<div class="tt-note">${note}</div>` : ''}`;

    switch (hit.layer) {
      case 'input':
        return card(`Input pixel (${hit.r}, ${hit.c})`, [row('Value', fmt(hit.value))], 'Brightness after preprocessing, 0 = background, 1 = ink.');

      case 'conv1Kernels':
      case 'conv2Kernels': {
        const p = hit.panel;
        const layer = hit.layer === 'conv1Kernels' ? 'conv1' : 'conv2';
        const where = layer === 'conv2' ? ` · input channel ${p.inChannel}` : '';
        return card(`${layer} filter ${p.channel}${where}`, [row(`Weight (${hit.r}, ${hit.c})`, fmt(hit.value)), row('Filter bias', fmt(W[layer].bias[p.channel]))],
          'Learned weight. Blue = negative, white = 0, red = positive.');
      }

      case 'conv1':
      case 'conv2': {
        const ch = hit.panel.channel;
        const src = hit.layer === 'conv1' ? { act: R.input, grid: grids.input } : { act: R.pool1, grid: grids.pool1 };
        const inbound = convInbound(W[hit.layer], src.act, hit.r, hit.c, ch);
        const maxC = Math.max(...inbound.map((e) => Math.abs(e.contribution)), 1e-9);
        const to = anchor(hit, 'front');
        let sum = 0;
        for (const e of inbound) {
          sum += e.contribution;
          const from = anchor(src.grid.cell(hit.layer === 'conv1' ? 0 : e.ci, e.r, e.c), 'front');
          // Zero inputs still get a muted line so the whole window stays visible.
          if (e.contribution === 0) focusLinks.push(from, to, MUTED, 0.5);
          else pushLink(focusLinks, from, to, e.contribution, maxC, 0.25);
        }
        const bias = W[hit.layer].bias[ch];
        const n = hit.layer === 'conv1' ? '3×3 pixels' : '3×3×8 inputs';
        return card(`${hit.layer} · filter ${ch} · (${hit.r}, ${hit.c})`, [
          row('Σ w·x', fmt(sum)), row('+ bias', fmt(bias)), row('= output', fmt(hit.value)), row('After ReLU', fmt(Math.max(0, hit.value))),
        ], `Weighted sum over ${n}. Lines show each input's w·x: red adds, blue subtracts.`);
      }

      case 'pool1':
      case 'pool2': {
        const ch = hit.panel.channel;
        const [reluName, convGrid] = hit.layer === 'pool1' ? ['relu1', grids.conv1] : ['relu2', grids.conv2];
        const to = anchor(hit, 'front');
        const cells = poolInbound(R[reluName], hit.r, hit.c, ch);
        for (const e of cells) {
          focusLinks.push(anchor(convGrid.cell(ch, e.r, e.c), 'front'), to, e.isMax ? ACCENT : MUTED, e.isMax ? 1 : 0.6);
        }
        const conv = hit.layer === 'pool1' ? 'conv1' : 'conv2';
        return card(`${hit.layer} · map ${ch} · (${hit.r}, ${hit.c})`, [
          row('Value', fmt(hit.value)), ...cells.map((e) => row(`ReLU(${conv}) (${e.r}, ${e.c})${e.isMax ? ' ★' : ''}`, fmt(e.value))),
        ], 'Max of the 2×2 window after ReLU, so negatives are gone.');
      }

      case 'flatten': {
        const { r, c, ch } = unflatten(hit.index);
        focusLinks.push(anchor(grids.pool2.cell(ch, r, c), 'front'), anchor(hit, 'left'), ACCENT, 1);
        return card(`flatten[${hit.index}]`, [row('Value', fmt(hit.value)), row('From pool2', `map ${ch} · (${r}, ${c})`)],
          'Flatten reads the 7×7×16 block row by row, channel fastest.');
      }

      case 'dense1': {
        const j = hit.index;
        const set = denseTopK(W.dense1, R.flatten.data, j, k);
        const to = anchor(hit, 'left');
        const max = maxContribution([set]);
        for (const e of set) pushLink(focusLinks, flattenAnchor(e.i), to, e.contribution, max, 0.35);
        linkDim.links1 = 0.18;
        return card(`dense1 · neuron ${j}`, [
          row('Weighted sum z', fmt(R.dense1.data[j])), row('Bias', fmt(W.dense1.bias[j])), row('After ReLU', fmt(R.relu3.data[j])),
        ], `Lines: the ${k} strongest of 784 inputs by |w·x|.`);
      }

      case 'output': {
        const j = hit.index;
        const set = denseTopK(W.dense2, R.relu3.data, j, k);
        const to = anchor(hit, 'left');
        const max = maxContribution([set]);
        for (const e of set) pushLink(focusLinks, anchor(dense.cell(e.i), 'right'), to, e.contribution, max, 0.35);
        linkDim.links2 = 0.18;
        return card(`Digit ${j}`, [row('Probability', `${(R.softmax.data[j] * 100).toFixed(2)}%`), row('Logit', fmt(R.dense2.data[j])), row('Bias', fmt(W.dense2.bias[j]))],
          `Softmax turns the 10 logits into probabilities. Lines: the ${k} strongest of 64 inputs.`);
      }
      default:
        return '';
    }
  }

  // --- camera --------------------------------------------------------------------------
  let fly = null;
  controls.addEventListener('start', () => {
    fly = null;
  });

  function viewFor(box, azimuth, elevation, margin = 1.15) {
    const size = box.getSize(new Vector3());
    const center = box.getCenter(new Vector3());
    const vfov = (camera.fov * Math.PI) / 180;
    const hfov = 2 * Math.atan(Math.tan(vfov / 2) * camera.aspect);
    const dist = margin * Math.max(size.y / 2 / Math.tan(vfov / 2), size.x / 2 / Math.tan(hfov / 2)) + size.z;
    const dir = new Vector3(Math.sin(azimuth) * Math.cos(elevation), Math.sin(elevation), Math.cos(azimuth) * Math.cos(elevation));
    // The estimate above ignores perspective: viewed at an angle, the near end of a wide box
    // looms larger. Back off until every corner projects inside the frame.
    const probe = camera.clone();
    const corners = [];
    for (const x of [box.min.x, box.max.x]) for (const y of [box.min.y, box.max.y]) for (const z of [box.min.z, box.max.z]) corners.push(new Vector3(x, y, z));
    const v = new Vector3();
    let d = dist;
    for (let i = 0; i < 30; i++) {
      probe.position.copy(center).addScaledVector(dir, d);
      probe.lookAt(center);
      probe.updateMatrixWorld();
      const fits = corners.every((c) => {
        v.copy(c).project(probe);
        return Math.abs(v.x) <= 0.96 && Math.abs(v.y) <= 0.96 && v.z < 1;
      });
      if (fits) break;
      d *= 1.08;
    }
    return { position: center.clone().addScaledVector(dir, d), target: center };
  }

  function stationBox(s) {
    return new Box3(new Vector3(s.left - 2, -s.height / 2 - 3, 0), new Vector3(s.right + 2, s.height / 2 + 10, 2));
  }

  function overviewView() {
    // In a portrait viewport the full 500-unit-wide network would be a thin strip,
    // so start on the first layers instead; orbit/zoom/layer buttons reach the rest.
    if (camera.aspect < 1) {
      const box = stationBox(S.input).union(stationBox(S.pool1));
      return viewFor(box, -0.3, 0.12, 1.0);
    }
    const box = new Box3(new Vector3(0, -75, 0), new Vector3(L.width, 80, 2));
    return viewFor(box, -0.32, 0.2, 1.0);
  }

  function flyTo({ position, target }, seconds = FLY_SECONDS) {
    if (seconds <= 0) {
      camera.position.copy(position);
      controls.target.copy(target);
      controls.update();
      invalidate();
      return Promise.resolve();
    }
    return new Promise((done) => {
      fly = { from: camera.position.clone(), to: position, fromT: controls.target.clone(), toT: target, t: 0, seconds, done };
    });
  }

  function focus(id, seconds) {
    onFocus?.(id);
    const s = S[id];
    // Columns are tall and narrow; include some neighbours so the view isn't a sliver.
    const box = stationBox(s);
    if (s.width < 40) box.expandByVector(new Vector3((40 - s.width) / 2, 0, 0));
    return flyTo(viewFor(box, -0.38, 0.12), seconds);
  }

  function overview(seconds) {
    onFocus?.(null);
    return flyTo(overviewView(), seconds);
  }

  /** Frame several stations at once (e.g. input + conv1 for the kernel slide). */
  function focusRange(ids, seconds) {
    const box = new Box3();
    for (const id of ids) box.union(stationBox(S[id]));
    return flyTo(viewFor(box, -0.3, 0.1, 1.05), seconds);
  }

  // --- wheel, trackpad and pinch -----------------------------------------------------
  // Captured on the container so OrbitControls' own wheel zoom never runs.
  //  - pinch on a trackpad arrives as ctrl+wheel (Chrome, Edge, Firefox): zoom at the cursor
  //  - two-finger scroll on a trackpad: slide the view
  //  - a mouse wheel (line mode, or whole 100/120-px notches): zoom at the cursor
  //  - Safari sends gesture* events for pinch instead
  const tmpPoint = new Vector3();
  const tmpOffset = new Vector3();
  let lastTrackpadAt = -Infinity;

  /** Zoom by `factor` (<1 = closer), keeping the point under (clientX, clientY) fixed on screen. */
  function zoomAt(factor, clientX, clientY) {
    fly = null;
    const dist = camera.position.distanceTo(controls.target);
    factor = clamp(factor, controls.minDistance / dist, controls.maxDistance / dist);
    let anchorPoint = controls.target;
    if (clientX !== undefined) {
      const rect = renderer.domElement.getBoundingClientRect();
      ndc.set(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
      raycaster.setFromCamera(ndc, camera);
      plane.constant = 0;
      if (raycaster.ray.intersectPlane(plane, tmpPoint) && tmpPoint.distanceTo(camera.position) < controls.maxDistance * 2) {
        anchorPoint = tmpPoint;
      }
    }
    const p = anchorPoint.clone();
    camera.position.sub(p).multiplyScalar(factor).add(p);
    controls.target.sub(p).multiplyScalar(factor).add(p);
    controls.update();
    invalidate();
  }

  /** Slide the view by a screen-space amount in pixels. */
  function panBy(dxPx, dyPx) {
    fly = null;
    const dist = camera.position.distanceTo(controls.target);
    const perPx = (2 * dist * Math.tan((camera.fov * Math.PI) / 360)) / renderer.domElement.clientHeight;
    tmpOffset.setFromMatrixColumn(camera.matrix, 0).multiplyScalar(dxPx * perPx);
    const up = new Vector3().setFromMatrixColumn(camera.matrix, 1).multiplyScalar(-dyPx * perPx);
    tmpOffset.add(up);
    camera.position.add(tmpOffset);
    controls.target.add(tmpOffset);
    controls.update();
    invalidate();
  }

  container.addEventListener('wheel', (e) => {
    e.preventDefault();
    e.stopPropagation();
    const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1;
    const dx = e.deltaX * unit;
    const dy = e.deltaY * unit;
    if (e.ctrlKey || e.metaKey) {
      zoomAt(Math.exp(clamp(dy, -50, 50) * 0.01), e.clientX, e.clientY); // pinch
      return;
    }
    const now = performance.now();
    const trackpadLike = e.deltaMode === 0 && (dx !== 0 || !Number.isInteger(dy) || Math.abs(dy) < 50);
    if (trackpadLike) lastTrackpadAt = now;
    if (trackpadLike || now - lastTrackpadAt < 400) panBy(dx, dy);
    else zoomAt(Math.exp(clamp(dy, -240, 240) * 0.0015), e.clientX, e.clientY); // mouse wheel notch ≈ 16%
  }, { passive: false, capture: true });

  let gestureScale = 1;
  container.addEventListener('gesturestart', (e) => {
    e.preventDefault();
    gestureScale = 1;
  });
  container.addEventListener('gesturechange', (e) => {
    e.preventDefault();
    zoomAt(gestureScale / e.scale, e.clientX, e.clientY);
    gestureScale = e.scale;
  });

  // --- render loop -------------------------------------------------------------------------
  let flattenAnim = null;
  const anims = new Set(); // step animations: { t, seconds, frame(t), resolve }
  let paused = false; // freezes the flatten/slide animations (playback pause)
  let speed = 1;
  let last = performance.now();
  let renders = 0;

  renderer.setAnimationLoop((now) => {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    let active = false;
    if (fly) {
      fly.t = Math.min(1, fly.t + dt / fly.seconds);
      const e = easeInOut(fly.t);
      camera.position.lerpVectors(fly.from, fly.to, e);
      controls.target.lerpVectors(fly.fromT, fly.toT, e);
      if (fly.t >= 1) {
        fly.done();
        fly = null;
      }
      active = true;
    }
    if (controls.update()) active = true;
    const adt = paused ? 0 : dt * speed;
    if (flattenAnim && adt > 0) {
      flattenAnim.t = Math.min(1, flattenAnim.t + adt / FLATTEN_SECONDS);
      flatten.setProgress(flattenAnim.t);
      links1.lines.visible = flattenAnim.t >= 1;
      if (flattenAnim.t >= 1) {
        flattenAnim.done();
        flattenAnim = null;
      }
      active = true;
    }
    if (anims.size && adt > 0) {
      for (const a of [...anims]) {
        a.t = Math.min(1, a.t + adt / a.seconds);
        a.frame(a.t);
        if (a.t >= 1) {
          anims.delete(a);
          a.resolve(true);
        }
      }
      active = true;
    }
    for (const id in levels) {
      const diff = targets[id] - levels[id];
      if (Math.abs(diff) < 0.002) {
        levels[id] = targets[id];
        continue;
      }
      levels[id] += diff * Math.min(1, dt * 5);
      active = true;
    }
    if (needsRender || active) {
      applyLevels();
      renderer.render(scene, camera);
      labelRenderer.render(scene, camera);
      needsRender = false;
      renders++;
    }
  });

  /** Run `frame(t)` for t: 0 -> 1 over `seconds` of animation time. Resolves false if cancelled. */
  function animate(seconds, frame) {
    return new Promise((resolve) => {
      const a = { t: 0, seconds: Math.max(0.001, seconds), frame, resolve };
      anims.add(a);
      frame(0);
      invalidate();
    });
  }

  function runFlatten() {
    return new Promise((done) => {
      flattenAnim = { t: 0, done };
    });
  }

  function playFlatten() {
    cancelAnimations();
    hovered = null;
    pinned = null;
    refreshFocus();
    return runFlatten();
  }

  /** Stop every running animation and restore the normal display. */
  function cancelAnimations() {
    if (flattenAnim) {
      const { done } = flattenAnim;
      flattenAnim = null;
      flatten.setProgress(1);
      links1.lines.visible = true;
      done();
    }
    for (const a of anims) a.resolve(false);
    anims.clear();
    steps?.cancel();
    invalidate();
  }

  function setLinkDim(v) {
    linkDim.links1 = v;
    linkDim.links2 = v;
    invalidate();
  }

  const steps = createSteps({
    scene, S, W, grids, mods, neuronMods, dense, output, flatten, focusLinks, outline, flyers,
    getResult: () => result,
    getK: () => k,
    setCaption(html) {
      caption.hidden = !html;
      if (html) caption.innerHTML = html;
    },
    setProbOverride(p) {
      probOverride = p;
      refreshNeurons();
    },
    setLinkDim, showBox, hideBoxes, setFlyer, refreshGrid, refreshNeurons, refreshAll,
    animate, runFlatten, focus, focusRange, anchor, pushLink, flattenAnchor,
  });

  // --- sizing ---------------------------------------------------------------------------------
  function resize() {
    const w = Math.max(1, container.clientWidth);
    const h = Math.max(1, container.clientHeight);
    renderer.setSize(w, h);
    labelRenderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    container.classList.toggle('compact', w < 640);
    invalidate();
  }
  const ro = new ResizeObserver(resize);
  ro.observe(container);
  resize();
  update(null);
  overview(0);

  function setTopK(next) {
    k = Math.max(1, Math.min(MAX_K, next | 0));
    updateLinks();
    refreshFocus();
  }

  function stationOf(layer) {
    return { conv1Kernels: 'conv1', conv2Kernels: 'conv2' }[layer] ?? layer;
  }

  return {
    update,
    setTopK,
    playFlatten,
    /** Fly to a station and play its step animation (see viz/steps.js). */
    playStep: (id, opts) => steps.playStep(id, opts),
    cancelAnimations,
    focus,
    focusRange,
    overview,
    setLit,
    setSpeed(s) {
      speed = s;
    },
    setDragMode,
    /** Zoom around the view center (buttons / keyboard). */
    zoomBy(factor) {
      zoomAt(factor);
    },
    setPaused(p) {
      paused = p;
    },
    stations: L.stations.map((s) => s.id),
    layout: L,
    dispose() {
      renderer.setAnimationLoop(null);
      ro.disconnect();
      controls.dispose();
      renderer.dispose();
    },
    /** Test/debug helpers: screen position of a unit, and render statistics. */
    debug: {
      screenOf(layer, a, b, c) {
        const hit = resolve({ layer, panelIndex: b === undefined ? undefined : a, r: b, c, index: b === undefined ? a : undefined });
        const v = new Vector3(...hit.center);
        v.z += hit.size[2] / 2;
        v.project(camera);
        const rect = canvas.getBoundingClientRect();
        return { x: rect.left + ((v.x + 1) / 2) * rect.width, y: rect.top + ((1 - v.y) / 2) * rect.height };
      },
      levels: () => ({ ...levels }),
      cameraState: () => ({
        position: camera.position.toArray(),
        target: controls.target.toArray(),
        distance: camera.position.distanceTo(controls.target),
        direction: camera.position.clone().sub(controls.target).normalize().toArray(),
      }),
      /** Client position of a world point. */
      project([x, y, z]) {
        const v = new Vector3(x, y, z).project(camera);
        const rect = renderer.domElement.getBoundingClientRect();
        return { x: rect.left + ((v.x + 1) / 2) * rect.width, y: rect.top + ((1 - v.y) / 2) * rect.height };
      },
      /** World point on the z = 0 plane under a client position. */
      worldAt(clientX, clientY) {
        const rect = renderer.domElement.getBoundingClientRect();
        ndc.set(((clientX - rect.left) / rect.width) * 2 - 1, -((clientY - rect.top) / rect.height) * 2 + 1);
        raycaster.setFromCamera(ndc, camera);
        plane.constant = 0;
        return raycaster.ray.intersectPlane(plane, new Vector3())?.toArray() ?? null;
      },
      targets: () => ({ ...targets }),
      sliding: () => steps.phase() === 'slide',
      phase: () => steps.phase(),
      /** What the step animations are currently showing (for tests). */
      anim: () => ({
        phase: steps.phase(),
        flyers: flyers.count,
        pulses: steps.pulseCount(),
        windows: boxes.filter((b) => b.visible).length,
        shownProbs: probOverride ? Array.from(probOverride) : null,
        conv1Negatives: grids.conv1.values.reduce((n, v) => n + (v < 0 ? 1 : 0), 0),
      }),
      stats: () => ({ renders, calls: renderer.info.render.calls, triangles: renderer.info.render.triangles, links: links1.n + links2.n }),
      camera,
      controls,
    },
  };
}

// --- flatten column -------------------------------------------------------------------------

/** 784 bars that can animate from the 7x7x16 pool2 block into a vertical column. */
class FlattenColumn {
  constructor(station) {
    this.s = station;
    this.mesh = instanced(BOX, 784);
    this.mesh.name = 'flatten';
    this.values = new Float32Array(784);
    this.depth = new Float32Array(784);
    this.progress = 1;
  }

  update(data, max) {
    const col = this.mesh.instanceColor.array;
    const lut = LUTS.sequential;
    for (let i = 0; i < 784; i++) {
      const v = data ? data[i] : 0;
      const n = max > 0 ? v / max : 0;
      this.values[i] = v;
      this.depth[i] = 0.2 + n * 1.6;
      const slot = lutIndex(v, max, false);
      col.set([lut[slot * 3], lut[slot * 3 + 1], lut[slot * 3 + 2]], i * 3);
    }
    this.mesh.instanceColor.needsUpdate = true;
    this.setProgress(this.progress);
  }

  /** t = 0: 7x7x16 block; t = 1: column. Cells leave the block in flatten order. */
  setProgress(t) {
    this.progress = t;
    const { column: col, block } = this.s;
    const p = block.pitch;
    const mat = this.mesh.instanceMatrix.array;
    for (let i = 0; i < 784; i++) {
      const e = easeInOut(clamp((t - (i / 783) * 0.6) / 0.4, 0, 1));
      const { r, c, ch } = unflatten(i);
      const d = this.depth[i];
      const bx = block.x + (c - 3) * p;
      const by = (3 - r) * p;
      const bz = (7.5 - ch) * p;
      const cy = col.top - i * col.pitch;
      writeMatrix(
        mat, i,
        lerp(bx, col.x, e), lerp(by, cy, e), lerp(bz, d / 2, e),
        lerp(p * 0.85, col.barW, e), lerp(p * 0.85, col.barH, e), lerp(p * 0.85, d, e),
      );
    }
    this.mesh.instanceMatrix.needsUpdate = true;
  }

  pick(x, y) {
    if (this.progress < 1) return null;
    const { column: col } = this.s;
    if (Math.abs(x - col.x) > col.barW / 2 + 0.4) return null;
    const i = Math.round((col.top - y) / col.pitch);
    return i >= 0 && i < 784 ? this.cell(i) : null;
  }

  cell(i) {
    const { column: col } = this.s;
    return {
      layer: 'flatten', index: i, instance: i, value: this.values[i],
      center: [col.x, col.top - i * col.pitch, this.depth[i] / 2],
      size: [col.barW, col.barH, this.depth[i]],
    };
  }
}

// --- helpers -------------------------------------------------------------------------------------

/** Where a connection line attaches to a unit. */
function anchor(hit, side) {
  const [x, y, z] = hit.center;
  if (hit.radius !== undefined) {
    return side === 'right' ? [x + hit.radius, y, z] : [x - hit.radius, y, z];
  }
  const [sx, , sz] = hit.size;
  if (side === 'front') return [x, y, z + sz / 2];
  return side === 'right' ? [x + sx / 2, y, z] : [x - sx / 2, y, z];
}

function labelEl(className, html) {
  const el = document.createElement('div');
  el.className = className;
  el.innerHTML = html;
  return el;
}

function buildLegend() {
  const el = document.createElement('div');
  el.className = 'scene-legend';
  const item = (name, lo, mid, hi, caption) =>
    `<div class="lg-item"><div class="lg-caption">${caption}</div><div class="lg-ramp" style="background:${cssGradient(name)}"></div>` +
    `<div class="lg-ticks"><span>${lo}</span><span>${mid}</span><span>${hi}</span></div></div>`;
  el.innerHTML =
    item('sequential', '0', '', 'max', 'Activation (after ReLU)') +
    item('diverging', '−', '0', '+', 'Conv output (before ReLU)') +
    item('kernel', '−', '0', '+', 'Filter weights') +
    '<div class="lg-item lg-links"><div class="lg-caption">Connections: w·x</div><div><i class="lg-pos"></i> adds <i class="lg-neg"></i> subtracts</div></div>';
  return el;
}

function maxAbs(a) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]));
  return m;
}

function maxContribution(sets) {
  let m = 0;
  for (const set of sets) for (const e of set) m = Math.max(m, Math.abs(e.contribution));
  return m;
}

function argmax(a) {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
  return best;
}

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const lerp = (a, b, t) => a + (b - a) * t;
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
