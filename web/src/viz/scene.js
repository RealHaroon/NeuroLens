// The 3D network: builds every station once, then each new forward pass only
// rewrites instance colors/matrices and the top-k connection buffers.
//
// Rendering is on demand: a frame is drawn only when the camera moves, data
// changes, or an animation is running, which keeps an idle laptop cool.

import {
  AmbientLight, Box3, BoxGeometry, Color, DirectionalLight, EdgesGeometry, LineBasicMaterial,
  LineSegments, PerspectiveCamera, Plane, Raycaster, Scene, Vector2, Vector3, WebGLRenderer,
} from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DObject, CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';
import { buildLayout } from './layout.js';
import { BOX, GridLayer, instanced, LineBatch, NeuronLayer, writeMatrix } from './primitives.js';
import { ACCENT, cssGradient, LUTS, lutIndex, MUTED, SCENE_BG, SIGN_COLORS } from './colors.js';
import { convInbound, denseTopK, poolInbound, unflatten } from './connections.js';

const MAX_K = 32;
const FLATTEN_SECONDS = 2.6;
const FLY_SECONDS = 0.9;

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

export function createNetworkScene(container, inspector) {
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

  // --- labels --------------------------------------------------------------------
  for (const s of L.stations) {
    const [title, sub] = STATION_LABELS[s.id];
    const el = labelEl('scene-label station', `<strong>${title}</strong><span>${sub}</span>`);
    const obj = new CSS2DObject(el);
    obj.position.set(s.center[0], s.height / 2 + 7, 0);
    scene.add(obj);
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

  // --- data updates --------------------------------------------------------------
  let result = null;
  let k = 8;

  function update(next) {
    result = next;
    const zero = null;
    grids.input.update(result?.input.data ?? zero, 1);
    grids.conv1.update(result?.conv1.data ?? zero, result ? maxAbs(result.conv1.data) : 0);
    grids.pool1.update(result?.pool1.data ?? zero, result ? maxAbs(result.pool1.data) : 0);
    grids.conv2.update(result?.conv2.data ?? zero, result ? maxAbs(result.conv2.data) : 0);
    grids.pool2.update(result?.pool2.data ?? zero, result ? maxAbs(result.pool2.data) : 0);
    flatten.update(result?.flatten.data ?? zero, result ? maxAbs(result.flatten.data) : 0);
    dense.update(result?.relu3.data ?? zero, result ? maxAbs(result.relu3.data) : 0);
    output.update(result?.softmax.data ?? zero, 1);
    updateBars();
    updateLinks();
    refreshFocus();
    invalidate();
  }

  function updateBars() {
    const probs = result?.softmax.data;
    const top = probs ? argmax(probs) : -1;
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
    if (e.buttons) return; // orbiting
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
    links1.material.opacity = 1;
    links2.material.opacity = 1;
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
        links1.material.opacity = 0.18;
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
        links2.material.opacity = 0.18;
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
    return { position: center.clone().addScaledVector(dir, dist), target: center };
  }

  function stationBox(s) {
    return new Box3(new Vector3(s.left - 2, -s.height / 2 - 3, 0), new Vector3(s.right + 2, s.height / 2 + 10, 2));
  }

  function overviewView() {
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
    const s = S[id];
    // Columns are tall and narrow; include some neighbours so the view isn't a sliver.
    const box = stationBox(s);
    if (s.width < 40) box.expandByVector(new Vector3((40 - s.width) / 2, 0, 0));
    return flyTo(viewFor(box, -0.38, 0.12), seconds);
  }

  function overview(seconds) {
    return flyTo(overviewView(), seconds);
  }

  // --- render loop -------------------------------------------------------------------------
  let flattenAnim = null;
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
    if (flattenAnim) {
      flattenAnim.t = Math.min(1, flattenAnim.t + dt / FLATTEN_SECONDS);
      flatten.setProgress(flattenAnim.t);
      links1.lines.visible = flattenAnim.t >= 1;
      if (flattenAnim.t >= 1) {
        flattenAnim.done();
        flattenAnim = null;
      }
      active = true;
    }
    if (needsRender || active) {
      renderer.render(scene, camera);
      labelRenderer.render(scene, camera);
      needsRender = false;
      renders++;
    }
  });

  function playFlatten() {
    flattenAnim?.done();
    hovered = null;
    pinned = null;
    refreshFocus();
    return new Promise((done) => {
      flattenAnim = { t: 0, done };
    });
  }

  // --- sizing ---------------------------------------------------------------------------------
  function resize() {
    const w = Math.max(1, container.clientWidth);
    const h = Math.max(1, container.clientHeight);
    renderer.setSize(w, h);
    labelRenderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
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
    focus,
    overview,
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
