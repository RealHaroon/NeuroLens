// Per-layer animations for the forward-pass playback. Each step is a short,
// choreographed sequence that shows what the layer computes:
//
//   input, conv1, conv2  build-up wave (cells rise left to right), or a sliding
//                        kernel when "Kernel slide" is on (3x3 for conv1, 3x3x8 for conv2)
//   pool1, pool2         ReLU (negatives sink to 0), then a 2x2 max-pool window:
//                        the first row slowly with the max cell flying over, then fast
//   flatten              the existing block-to-column unroll
//   dense1               pulses travel along the top-k links; neurons swell as they arrive
//   output               pulses, then softmax "race" (the distribution sharpens), winner bounce
//
// Everything runs on the scene's animation clock (pause/speed aware). A new
// step or cancel() bumps the token, so an interrupted sequence stops at its
// next await and reset() restores the normal display.

import { AdditiveBlending, BufferAttribute, BufferGeometry, CanvasTexture, DynamicDrawUsage, Points, PointsMaterial } from 'three';
import { convInbound, denseTopK, poolInbound } from './connections.js';
import { MUTED, SIGN_COLORS } from './colors.js';

const MAX_PULSES = 64 * 32 + 10 * 32;

export function createSteps(ctx) {
  const { S, W, grids, mods, neuronMods } = ctx;
  let token = 0;
  let phase = null;
  const reducedMotion = typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;

  // --- signal pulses ---------------------------------------------------------------
  const pulsePos = new Float32Array(MAX_PULSES * 3);
  const pulseCol = new Float32Array(MAX_PULSES * 4);
  const pulseGeom = new BufferGeometry();
  pulseGeom.setAttribute('position', new BufferAttribute(pulsePos, 3).setUsage(DynamicDrawUsage));
  pulseGeom.setAttribute('color', new BufferAttribute(pulseCol, 4).setUsage(DynamicDrawUsage));
  pulseGeom.setDrawRange(0, 0);
  const pulses = new Points(pulseGeom, new PointsMaterial({
    size: 4.5, map: dotTexture(), vertexColors: true, transparent: true, depthWrite: false, blending: AdditiveBlending,
  }));
  pulses.frustumCulled = false;
  ctx.scene.add(pulses);

  // --- helpers -----------------------------------------------------------------------
  const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
  const smooth = (t) => t * t * (3 - 2 * t);
  const clamp01 = (v) => Math.min(1, Math.max(0, v));
  const R = () => ctx.getResult();

  function setPhase(p, html = null) {
    phase = p;
    ctx.setCaption(html);
  }

  /** Restore the normal display (called on cancel and at the start of every step). */
  function reset() {
    for (const name of Object.keys(mods)) delete mods[name];
    neuronMods.dense1 = null;
    neuronMods.output = null;
    ctx.setProbOverride(null);
    ctx.setLinkDim(1);
    ctx.hideBoxes();
    ctx.flyers.count = 0;
    ctx.outline.visible = false;
    ctx.focusLinks.clear();
    ctx.focusLinks.commit();
    pulseGeom.setDrawRange(0, 0);
    setPhase(null);
    ctx.refreshAll();
  }

  function cancel() {
    token++;
    reset();
  }

  // --- build-up wave ---------------------------------------------------------------------
  async function wave(names, station, seconds, alive) {
    const s = S[station];
    const band = 0.35;
    setPhase('wave');
    const ok = await ctx.animate(seconds, (t) => {
      const w = t * (1 + band);
      const reveal = (x) => smooth(clamp01((w - (x - s.left) / s.width) / band));
      for (const n of names) {
        mods[n] = { reveal };
        ctx.refreshGrid(n);
      }
    });
    if (!alive()) return false;
    for (const n of names) delete mods[n];
    for (const n of names) ctx.refreshGrid(n);
    setPhase(null);
    return ok;
  }

  // --- ReLU: negatives sink to zero ------------------------------------------------------------
  async function relu(conv, alive) {
    const data = R()[conv].data;
    let neg = 0;
    for (const v of data) if (v < 0) neg++;
    setPhase('relu', `<b>ReLU</b>: every negative value (blue) becomes 0 · ${neg.toLocaleString()} of ${data.length.toLocaleString()} values in ${conv}`);
    const ok = await ctx.animate(1.3, (t) => {
      mods[conv] = { reluT: ease(t) };
      ctx.refreshGrid(conv);
    });
    return ok && alive();
  }

  // --- max-pool: window sweep with flying max cells -----------------------------------------------
  async function pool(conv, poolName, reluName, alive) {
    const [H, Wd, C] = R()[poolName].shape;
    const flyers = ctx.flyers;
    const showWindows = (r, c) => {
      for (let ch = 0; ch < C; ch++) {
        const a = grids[conv].cell(ch, 2 * r, 2 * c);
        const b = grids[conv].cell(ch, 2 * r + 1, 2 * c + 1);
        const pitch = grids[conv].panels[ch].pitch;
        ctx.showBox(ch, [(a.center[0] + b.center[0]) / 2, (a.center[1] + b.center[1]) / 2, 1.2], [2 * pitch + 0.15, 2 * pitch + 0.15, 2.6]);
      }
    };
    const pooled = R()[poolName].data;
    // Demonstrate on the row where the digit is (largest pooled activation), then the rest.
    const rowSum = new Float32Array(H);
    for (let i = 0; i < pooled.length; i++) rowSum[Math.floor(i / C / Wd)] += pooled[i];
    const row = rowSum.indexOf(Math.max(...rowSum));
    const order = [];
    for (let c = 0; c < Wd; c++) order.push(row * Wd + c);
    for (let i = 0; i < H * Wd; i++) if (Math.floor(i / Wd) !== row) order.push(i);
    const rank = new Int32Array(H * Wd);
    order.forEach((cell, i) => (rank[cell] = i));
    const maskPool = (n) => {
      mods[poolName] = { mask: { visible: (cell) => rank[cell] < n, C } };
      ctx.refreshGrid(poolName);
    };
    // The busiest map in that row, for the caption.
    let showMap = 0;
    let best = -1;
    for (let ch = 0; ch < C; ch++) {
      let sum = 0;
      for (let c = 0; c < Wd; c++) sum += pooled[(row * Wd + c) * C + ch];
      if (sum > best) [best, showMap] = [sum, ch];
    }

    // One row in detail: the max of every window flies to its pooled cell.
    const ok1 = await ctx.animate(Math.max(1.6, Wd * 0.16), (t) => {
      const f = Math.min(Wd - 1e-6, t * Wd);
      const c = Math.floor(f);
      const sub = f - c;
      showWindows(row, c);
      flyers.count = C;
      const R0 = R();
      for (let ch = 0; ch < C; ch++) {
        const win = poolInbound(R0[reluName], row, c, ch);
        const m = win.find((e) => e.isMax);
        const from = grids[conv].cell(ch, m.r, m.c);
        const to = grids[poolName].cell(ch, row, c);
        const e = ease(clamp01(sub / 0.85));
        const base = grids[conv].panels[ch].size;
        const size = base * (1 + 1.1 * Math.sin(Math.PI * e));
        ctx.setFlyer(ch,
          [from.center[0] + (to.center[0] - from.center[0]) * e, from.center[1] + (to.center[1] - from.center[1]) * e, 1.5 + Math.sin(Math.PI * e) * 6],
          size, m.value > 0 ? 1 : 0.3);
      }
      const win = poolInbound(R0[reluName], row, c, showMap);
      const vals = win.map((e) => e.value.toFixed(2)).join(', ');
      setPhase('pool', `<b>Max-pool 2×2</b>: keep the largest of each window · map ${showMap}, window (${row}, ${c}): max(${vals}) = <b>${Math.max(...win.map((e) => e.value)).toFixed(2)}</b>`);
      maskPool(c + (sub > 0.85 ? 1 : 0));
    });
    if (!ok1 || !alive()) return false;
    flyers.count = 0;

    // The rest, fast-forwarded.
    const total = H * Wd;
    const ok2 = await ctx.animate(1.4, (t) => {
      const n = Math.min(total, Wd + Math.floor(t * (total - Wd + 1)));
      const at = order[Math.min(total - 1, n)];
      showWindows(Math.floor(at / Wd), at % Wd);
      setPhase('pool', `<b>Max-pool 2×2</b>: ${n} of ${total} windows · ${H}×${Wd} per map, ${C} maps`);
      maskPool(n);
    });
    if (!alive()) return false;
    ctx.hideBoxes();
    delete mods[poolName];
    ctx.refreshGrid(poolName);
    setPhase(null);
    return ok2;
  }

  // --- kernel slide (conv1: 3x3 over the input; conv2: 3x3x8 over all pool1 maps) -------------------
  function strongestFilter(conv) {
    const { data, shape } = R()[conv];
    const C = shape[2];
    const sums = new Float32Array(C);
    data.forEach((v, i) => {
      if (v > 0) sums[i % C] += v;
    });
    return sums.indexOf(Math.max(...sums));
  }

  async function slide(conv, seconds, alive) {
    const R0 = R();
    const [H, Wd, C] = R0[conv].shape;
    const ch = strongestFilter(conv);
    const src = conv === 'conv1' ? { act: 'input', grid: grids.input, maps: 1 } : { act: 'pool1', grid: grids.pool1, maps: 8 };
    const kernelPanels = S[conv].layers[`${conv}Kernels`].filter((p) => p.channel === ch);
    const kb = boundsOf(kernelPanels);
    const total = H * Wd;
    setPhase('slide');
    const ok = await ctx.animate(seconds, (t) => {
      const R1 = R();
      const done = Math.min(total, Math.floor(t * (total + 1)));
      mods[conv] = { mask: { n: done, C, channels: new Set([ch]) } };
      ctx.refreshGrid(conv);

      const at = Math.min(total - 1, done);
      const r = Math.floor(at / Wd);
      const c = at % Wd;
      for (let m = 0; m < src.maps; m++) {
        const cell = src.grid.cell(m, r, c);
        const pitch = src.grid.panels[m].pitch;
        ctx.showBox(m, [cell.center[0], cell.center[1], 1], [3 * pitch + 0.05, 3 * pitch + 0.05, 2.4]);
      }
      ctx.showBox(src.maps, kb.center, kb.size);

      const out = grids[conv].cell(ch, r, c);
      ctx.outline.visible = true;
      ctx.outline.position.set(...out.center);
      ctx.outline.scale.set(out.size[0] * 1.15, out.size[1] * 1.15, out.size[2] * 1.15);
      ctx.focusLinks.clear();
      const inbound = convInbound(W[conv], R1[src.act], r, c, ch);
      const maxC = Math.max(...inbound.map((e) => Math.abs(e.contribution)), 1e-9);
      const to = ctx.anchor(out, 'front');
      for (const e of inbound) {
        const from = ctx.anchor(src.grid.cell(conv === 'conv1' ? 0 : e.ci, e.r, e.c), 'front');
        if (e.contribution === 0) ctx.focusLinks.push(from, to, MUTED, 0.4);
        else ctx.pushLink(ctx.focusLinks, from, to, e.contribution, maxC, 0.3);
      }
      ctx.focusLinks.commit();
      const v = R1[conv].data[at * C + ch];
      const what = conv === 'conv1' ? '3×3 pixels' : '3×3 window on all 8 maps (72 inputs)';
      ctx.setCaption(`Filter ${ch} at row ${r}, col ${c}: Σ w·x + b over ${what} = <b>${v.toFixed(3)}</b> → ReLU <b>${Math.max(0, v).toFixed(3)}</b>`);
    });
    if (!alive()) return false;
    reset();
    return ok;
  }

  // --- signal pulses along the top-k links ------------------------------------------------------
  async function pulse(layer, seconds, alive) {
    const R0 = R();
    const k = ctx.getK();
    const [w, input, nOut, fromAnchor, toLayer] = layer === 'dense1'
      ? [W.dense1, R0.flatten.data, 64, (i) => ctx.flattenAnchor(i), ctx.dense]
      : [W.dense2, R0.relu3.data, 10, (i) => ctx.anchor(ctx.dense.cell(i), 'right'), ctx.output];
    const nIn = w.kernelShape[0];
    const list = [];
    let max = 0;
    for (let j = 0; j < nOut; j++) {
      for (const e of denseTopK(w, input, j, k)) {
        list.push({ j, from: fromAnchor(e.i), to: ctx.anchor(toLayer.cell(j), 'left'), c: e.contribution, delay: (e.i / nIn) * 0.45 });
        max = Math.max(max, Math.abs(e.contribution));
      }
    }
    const travel = 0.55;
    const counts = new Float32Array(nOut);
    for (const p of list) counts[p.j]++;
    ctx.setLinkDim(0.3);
    if (layer === 'output') ctx.setProbOverride(new Float32Array(10)); // scores aren't probabilities yet
    const caption = layer === 'dense1'
      ? `<b>Dense</b>: each of the 64 neurons adds up w·x over all 784 inputs plus a bias, then ReLU · pulses = its ${k} strongest inputs`
      : `<b>Output</b>: each digit’s score (logit) is a weighted sum of the 64 dense neurons · pulses = its ${k} strongest inputs`;
    setPhase('pulses', caption);
    const arrived = new Float32Array(nOut);
    const ok = await ctx.animate(seconds, (t) => {
      const T = t * (0.45 + travel); // timeline in seconds-equivalents: delays, then travel
      arrived.fill(0);
      let n = 0;
      for (const p of list) {
        const u = (T - p.delay) / travel;
        if (u >= 1) arrived[p.j]++;
        if (u <= 0 || u >= 1) continue;
        const e = ease(u);
        pulsePos[n * 3] = p.from[0] + (p.to[0] - p.from[0]) * e;
        pulsePos[n * 3 + 1] = p.from[1] + (p.to[1] - p.from[1]) * e;
        pulsePos[n * 3 + 2] = p.from[2] + (p.to[2] - p.from[2]) * e + 0.5;
        const col = p.c >= 0 ? SIGN_COLORS.positive : SIGN_COLORS.negative;
        const a = (0.45 + 0.55 * (max > 0 ? Math.abs(p.c) / max : 0)) * Math.sin(Math.PI * u) ** 0.5;
        pulseCol.set([col.r, col.g, col.b, a], n * 4);
        n++;
      }
      pulseGeom.attributes.position.needsUpdate = true;
      pulseGeom.attributes.color.needsUpdate = true;
      pulseGeom.setDrawRange(0, n);
      neuronMods[layer] = (j) => 0.12 + 0.88 * smooth(counts[j] ? arrived[j] / counts[j] : 1);
      ctx.refreshNeurons();
    });
    if (!alive()) return false;
    pulseGeom.setDrawRange(0, 0);
    neuronMods[layer] = null;
    ctx.setLinkDim(1);
    ctx.refreshNeurons();
    setPhase(null);
    return ok;
  }

  // --- softmax race: probabilities sharpen from near-uniform to the real softmax ------------------
  async function race(alive) {
    const logits = R().dense2.data;
    const top = logits.indexOf(Math.max(...logits));
    const T0 = 30;
    neuronMods.output = null;
    const ok = await ctx.animate(2.4, (t) => {
      const T = T0 + (1 - T0) * ease(t);
      const z = Array.from(logits, (v) => v / T);
      const m = Math.max(...z);
      const ex = z.map((v) => Math.exp(v - m));
      const sum = ex.reduce((a, b) => a + b, 0);
      ctx.setProbOverride(Float32Array.from(ex, (v) => v / sum));
      setPhase('race', `<b>Softmax</b>: e<sup>z</sup> / Σe<sup>z</sup> turns the 10 logits into probabilities that sum to 100% · sharpening from equal odds: <b>${Math.round((1 - (T - 1) / (T0 - 1)) * 100)}%</b>`);
    });
    if (!alive()) return false;
    ctx.setProbOverride(null);
    const ok2 = await ctx.animate(0.7, (t) => {
      neuronMods.output = (i) => (i === top ? 1 + 0.45 * Math.sin(Math.PI * t) : 1);
      ctx.refreshNeurons();
      setPhase('race', `Prediction: <b>${top}</b> (${(R().softmax.data[top] * 100).toFixed(1)}%)`);
    });
    if (!alive()) return false;
    neuronMods.output = null;
    ctx.refreshNeurons();
    return ok && ok2;
  }

  // --- step entry point -------------------------------------------------------------------------
  /**
   * Fly to a station and play its animation. Resolves true if it ran to the end,
   * false if it was interrupted (another step, cancel, or the input was cleared).
   */
  async function playStep(id, { kernelSlide = false, flySeconds = 1 } = {}) {
    const my = ++token;
    reset();
    const alive = () => my === token && !!R();
    const go = async (promise) => {
      await promise;
      return alive();
    };
    if (reducedMotion || !R()) return go(ctx.focus(id, flySeconds));

    switch (id) {
      case 'input':
        return (await go(ctx.focus('input', flySeconds))) && wave(['input'], 'input', 1.0, alive);
      case 'conv1':
      case 'conv2': {
        if (kernelSlide) {
          const from = id === 'conv1' ? 'input' : 'pool1';
          if (!(await go(ctx.focusRange([from, id], flySeconds)))) return false;
          return slide(id, id === 'conv1' ? 8 : 7, alive);
        }
        if (!(await go(ctx.focus(id, flySeconds)))) return false;
        return wave([`${id}Kernels`, id], id, 1.3, alive);
      }
      case 'pool1':
      case 'pool2': {
        const [conv, relName] = id === 'pool1' ? ['conv1', 'relu1'] : ['conv2', 'relu2'];
        if (!(await go(ctx.focusRange([conv, id], flySeconds)))) return false;
        return (await relu(conv, alive)) && pool(conv, id, relName, alive);
      }
      case 'flatten':
        if (!(await go(ctx.focus('flatten', flySeconds)))) return false;
        setPhase('flatten', '<b>Flatten</b>: the 7×7×16 block is read row by row into one list of 784 numbers');
        await ctx.runFlatten();
        if (!alive()) return false;
        setPhase(null);
        return true;
      case 'dense1':
        if (!(await go(ctx.focusRange(['flatten', 'dense1'], flySeconds)))) return false;
        return pulse('dense1', 2.2, alive);
      case 'output':
        if (!(await go(ctx.focusRange(['dense1', 'output'], flySeconds)))) return false;
        return (await pulse('output', 1.6, alive)) && race(alive);
      default:
        return go(ctx.focus(id, flySeconds));
    }
  }

  return { playStep, cancel, reset, phase: () => phase, pulseCount: () => pulseGeom.drawRange.count };
}

function boundsOf(panels) {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const p of panels) {
    x0 = Math.min(x0, p.x0 - p.pitch / 2);
    x1 = Math.max(x1, p.x0 + (p.cols - 0.5) * p.pitch);
    y1 = Math.max(y1, p.y0 + p.pitch / 2);
    y0 = Math.min(y0, p.y0 - (p.rows - 0.5) * p.pitch);
  }
  return { center: [(x0 + x1) / 2, (y0 + y1) / 2, 0.6], size: [x1 - x0 + 0.3, y1 - y0 + 0.3, 1.4] };
}

/** Soft round sprite for the pulses. */
function dotTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, 'rgba(255,255,255,1)');
  grad.addColorStop(0.35, 'rgba(255,255,255,0.8)');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  return new CanvasTexture(c);
}

