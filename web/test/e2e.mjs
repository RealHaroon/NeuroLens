// End-to-end check of the production build in headless Chrome:
// draws digits with the mouse, uploads a dark-on-white image, and verifies the
// predictions, the per-layer activations, and that tensors don't leak.
//
// Usage:  npm run test:e2e          (CHROME_PATH overrides the browser binary)

import { mkdtempSync, writeFileSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build, preview } from 'vite';
import puppeteer from 'puppeteer-core';

const CHROME = process.env.CHROME_PATH ?? '/usr/bin/google-chrome';
const root = fileURLToPath(new URL('..', import.meta.url));

await build({ root, logLevel: 'warn' });
const server = await preview({ root, preview: { port: 4174, strictPort: true }, logLevel: 'warn' });
const url = server.resolvedUrls.local[0];

const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, // Use the real GPU when there is one (falls back to SwiftShader otherwise).
  args: ['--no-sandbox', '--use-angle=gl', '--enable-unsafe-swiftshader'] });
let failed = false;
const check = (ok, msg) => {
  if (!ok) failed = true;
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${msg}`);
};

try {
  const page = await browser.newPage();
  await page.setViewport({ width: 1280, height: 900 });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));

  await page.goto(url);
  await page.waitForFunction(() => window.neurolens?.ready, { timeout: 20000 });
  console.log(`Loaded ${url}`);

  const pad = await page.$('#drawpad');
  const box = await pad.boundingBox();
  const sx = box.width / 280;

  async function draw(strokes) {
    await page.click('#clear');
    for (const stroke of strokes) {
      await page.mouse.move(box.x + stroke[0][0] * sx, box.y + stroke[0][1] * sx);
      await page.mouse.down();
      for (const [x, y] of stroke.slice(1)) {
        await page.mouse.move(box.x + x * sx, box.y + y * sx, { steps: 12 });
      }
      await page.mouse.up();
    }
    await settle();
    return page.evaluate(() => {
      const r = window.neurolens.last;
      if (!r) return null;
      const p = Array.from(r.softmax.data);
      return { pred: p.indexOf(Math.max(...p)), conf: Math.max(...p) };
    });
  }

  async function settle() {
    // Wait two frames past the last scheduled inference.
    await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => setTimeout(r, 50)))));
  }

  const circle = Array.from({ length: 25 }, (_, i) => {
    const t = (i / 24) * Math.PI * 2;
    return [140 + 60 * Math.sin(t), 140 - 95 * Math.cos(t)];
  });
  const cases = [
    ['1', [[[150, 40], [135, 245]]]],
    ['7', [[[70, 55], [215, 55], [120, 240]]]],
    ['0', [circle]],
    ['4', [[[100, 40], [70, 160], [210, 160]], [[175, 70], [175, 250]]]],
  ];
  console.log('Drawing with the mouse:');
  for (const [digit, strokes] of cases) {
    const r = await draw(strokes);
    check(r && String(r.pred) === digit, `drew "${digit}" -> predicted ${r?.pred} (${(r?.conf * 100).toFixed(1)}%)`);
  }

  console.log('Activations exposed to the page:');
  const info = await page.evaluate(() => {
    const { last, inspector } = window.neurolens;
    return {
      stages: Object.fromEntries(Object.entries(last).map(([k, v]) => [k, v.shape.join('x')])),
      weights: Object.fromEntries(Object.entries(inspector.weights).map(([k, v]) => [k, v.kernelShape.join('x')])),
      reluNegatives: last.relu1.data.filter((v) => v < 0).length,
      convNegatives: last.conv1.data.filter((v) => v < 0).length,
      rows: document.querySelectorAll('#layers tbody tr').length,
    };
  });
  console.log(`    ${Object.entries(info.stages).map(([k, v]) => `${k}:${v}`).join('  ')}`);
  console.log(`    weights ${Object.entries(info.weights).map(([k, v]) => `${k}:${v}`).join('  ')}`);
  check(Object.keys(info.stages).length === 12, '12 stages returned from one forward pass');
  check(info.convNegatives > 0 && info.reluNegatives === 0, `conv1 has ${info.convNegatives} negatives, relu1 has 0`);
  check(info.rows === 12, 'layer table rendered');

  // Tensor leak check across many live-drawing passes.
  const before = await page.evaluate(() => document.getElementById('status').textContent);
  for (let i = 0; i < 3; i++) await draw(cases[i % cases.length][1]);
  const after = await page.evaluate(() => document.getElementById('status').textContent);
  const count = (s) => Number(/(\d+) tensors/.exec(s)?.[1]);
  check(count(before) === count(after), `tensor count stable while drawing (${count(before)} -> ${count(after)})`);

  // Upload: an MNIST digit rendered dark-on-white at 300x300, off-center.
  const parity = JSON.parse(readFileSync(new URL('./fixtures/parity.json', import.meta.url), 'utf8'));
  const dataUrl = await page.evaluate((b64) => {
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const img = new Float32Array(bytes.buffer).subarray(0, 784); // test image 0 (a 7)
    const c = document.createElement('canvas');
    c.width = c.height = 300;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#f2efe6';
    ctx.fillRect(0, 0, 300, 300);
    for (let i = 0; i < 784; i++) {
      const v = Math.round(242 - img[i] * 200);
      ctx.fillStyle = `rgb(${v},${v},${v + 10})`;
      ctx.fillRect(40 + (i % 28) * 7, 20 + Math.floor(i / 28) * 7, 7, 7);
    }
    return c.toDataURL('image/png');
  }, parity.images.data);
  const tmp = mkdtempSync(join(tmpdir(), 'neurolens-'));
  const file = join(tmp, 'digit7.png');
  writeFileSync(file, Buffer.from(dataUrl.split(',')[1], 'base64'));
  await (await page.$('#upload')).uploadFile(file);
  await page.waitForFunction(() => window.neurolens.last, { timeout: 5000 });
  await page.waitForFunction(() => window.neurolens.last?.softmax.data[7] > 0.5, { timeout: 5000 }).catch(() => {});
  const up = await page.evaluate(() => {
    const p = Array.from(window.neurolens.last.softmax.data);
    return { pred: p.indexOf(Math.max(...p)), conf: Math.max(...p) };
  });
  console.log('Upload:');
  check(up.pred === parity.labels[0], `dark-on-white upload -> predicted ${up.pred} (${(up.conf * 100).toFixed(1)}%), label ${parity.labels[0]}`);

  // --- 3D scene -------------------------------------------------------------
  console.log('3D scene:');
  await page.waitForFunction(() => window.neurolens.scene, { timeout: 20000 });
  await draw(cases[1][1]); // a 7
  const scene = (fn, ...args) => page.evaluate(fn, ...args);
  const stats = () => scene(() => window.neurolens.scene.debug.stats());
  const frame = () => scene(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
  let st = await stats();
  check(st.calls <= 20, `whole network in ${st.calls} draw calls (instanced), ${st.triangles.toLocaleString()} triangles`);
  check(st.links === 64 * 8 + 10 * 8, `top-8 connections drawn: ${st.links} lines (64×8 + 10×8)`);

  await scene(() => {
    const el = document.getElementById('topk');
    el.value = '3';
    el.dispatchEvent(new Event('input'));
  });
  st = await stats();
  check(st.links === 64 * 3 + 10 * 3, `k slider = 3 -> ${st.links} lines`);
  await scene(() => {
    const el = document.getElementById('topk');
    el.value = '8';
    el.dispatchEvent(new Event('input'));
  });

  async function hover(station, layer, args, expect) {
    await scene((id) => window.neurolens.scene.focus(id, 0), station);
    await frame();
    const pos = await scene((l, a) => window.neurolens.scene.debug.screenOf(l, ...a), layer, args);
    await page.mouse.move(pos.x, pos.y);
    await frame();
    const tip = await page.$eval('.scene-tooltip', (el) => (el.hidden ? '' : el.textContent));
    check(tip.includes(expect), `hover ${layer}(${args.join(', ')}) -> tooltip "${tip.slice(0, 60)}…"`);
    return pos;
  }
  await hover('conv1', 'conv1', [3, 10, 12], 'conv1 · filter 3 · (10, 12)');
  await hover('pool1', 'pool1', [5, 4, 6], 'pool1 · map 5 · (4, 6)');
  await hover('conv2', 'conv2', [12, 7, 3], 'conv2 · filter 12 · (7, 3)');
  await hover('flatten', 'flatten', [100], 'flatten[100]');
  await hover('dense1', 'dense1', [10], 'dense1 · neuron 10');
  const out7 = await hover('output', 'output', [7], 'Digit 7');

  await page.mouse.click(out7.x, out7.y);
  await page.mouse.move(out7.x - 300, out7.y - 200);
  await frame();
  const pinned = await page.$eval('.scene-tooltip', (el) => !el.hidden && el.classList.contains('pinned'));
  await page.keyboard.press('Escape');
  await frame();
  const cleared = await page.$eval('.scene-tooltip', (el) => el.hidden);
  check(pinned && cleared, 'click pins the tooltip, Escape clears it');

  const t0 = Date.now();
  await scene(() => window.neurolens.scene.playFlatten());
  const unrollMs = Date.now() - t0;
  check(unrollMs > 2000 && unrollMs < 6000, `flatten unroll animation ran (${(unrollMs / 1000).toFixed(1)} s)`);

  // --- Phase 4: playback, kernel slide, explanation panel ----------------------
  console.log('Playback:');
  const pstate = () => scene(() => window.neurolens.player.state());
  const text = (sel) => page.$eval(sel, (el) => el.textContent.trim());
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  await page.select('#speed', '4');
  await page.click('#play');
  await page.waitForFunction(() => window.neurolens.player.state().index === 2, { timeout: 10000 });
  const lit = await scene(() => window.neurolens.scene.debug.targets());
  const order = await scene(() => window.neurolens.scene.stations);
  const litOk = order.every((id, i) => (i <= 2 ? lit[id] === 1 : lit[id] < 0.2));
  check(litOk, `layers light up in order: at step 3 ${order.map((id) => (lit[id] === 1 ? id : '·')).join(' ')}`);
  check((await text('#play-label')) === 'Pause', 'play button turns into Pause while playing');
  check((await text('#about-title')).startsWith('ReLU + max-pool 1'), `about panel follows playback: "${await text('#about-title')}"`);

  await page.click('#play'); // pause
  const pausedAt = (await pstate()).index;
  await sleep(1500);
  const stillAt = (await pstate()).index;
  check(pausedAt === stillAt && (await text('#play-label')) === 'Resume', `pause holds the current layer (${pausedAt} -> ${stillAt})`);
  await page.click('#play'); // resume
  await page.waitForFunction((i) => window.neurolens.player.state().index > i, { timeout: 10000 }, stillAt);
  check(true, 'resume continues from where it paused');
  await page.click('#step-back');
  await frame();
  const back = await pstate();
  check(!back.playing && back.index === stillAt, `step back pauses and moves to layer ${back.index}`);
  await page.keyboard.press('ArrowRight');
  await frame();
  check((await pstate()).index === back.index + 1, 'ArrowRight steps forward');

  await page.click('#play');
  await page.waitForFunction(() => !window.neurolens.player.state().playing, { timeout: 30000 });
  const end = await pstate();
  const allLit = Object.values(await scene(() => window.neurolens.scene.debug.targets())).every((v) => v === 1);
  check(end.atEnd && allLit && (await text('#play-label')) === 'Replay forward pass', 'pass completes on the output with every layer lit');
  check((await text('#about-live')).startsWith('Prediction 7'), `live fact: "${await text('#about-live')}"`);

  console.log('Kernel slide:');
  await page.select('#speed', '1');
  await page.click('#kernel-slide');
  await scene(() => window.neurolens.player.goto('input'));
  await page.click('#step-forward'); // -> conv1 with the slide
  await page.waitForFunction(() => window.neurolens.scene.debug.sliding(), { timeout: 5000 });
  await sleep(2500);
  const cap1 = await text('.scene-caption');
  await sleep(1200);
  const cap2 = await text('.scene-caption');
  check(/^Filter \d at row \d+, col \d+/.test(cap1) && cap1 !== cap2, `window slides: "${cap1}" → "${cap2.slice(0, 26)}…"`);
  await page.waitForFunction(() => !window.neurolens.scene.debug.sliding(), { timeout: 15000 });
  check(await page.$eval('.scene-caption', (el) => el.hidden), 'slide finishes and restores the full feature maps');
  await page.click('#kernel-slide');

  console.log('About panel:');
  for (const [id, title, params] of [['conv1', 'Convolution 1', '80'], ['conv2', 'Convolution 2', '1,168'], ['dense1', 'Dense 1 + ReLU', '50,240'], ['output', 'Output + softmax', '650'], ['pool1', 'ReLU + max-pool 1', '0']]) {
    await page.click(`#stations button[data-station="${id}"]`);
    await frame();
    const t = await text('#about-title');
    const p = await text('#about-params');
    check(t === title && p === params, `${id}: "${t}", ${p} parameters`);
  }

  // Frame pacing while orbiting the full network (reported, not asserted:
  // it depends on the GPU, and SwiftShader software rendering is far slower).
  await scene(() => window.neurolens.scene.overview(0));
  const canvasBox = await (await page.$('.scene-canvas')).boundingBox();
  await scene(() => {
    window.__frames = [];
    const loop = (t) => {
      window.__frames.push(t);
      if (window.__frames.length < 400) requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  });
  const cx = canvasBox.x + canvasBox.width / 2;
  const cy = canvasBox.y + canvasBox.height / 2;
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  for (let i = 0; i < 60; i++) await page.mouse.move(cx + Math.sin(i / 8) * 200, cy + Math.cos(i / 8) * 60);
  await page.mouse.up();
  const perf = await scene(() => {
    const f = window.__frames;
    const d = f.slice(1).map((t, i) => t - f[i]).sort((a, b) => a - b);
    const gl = document.querySelector('.scene-canvas').getContext('webgl2');
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    return { median: d[Math.floor(d.length / 2)], p90: d[Math.floor(d.length * 0.9)], gpu: ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : '?' };
  });
  console.log(`  info frame time while orbiting: median ${perf.median.toFixed(1)} ms, p90 ${perf.p90.toFixed(1)} ms (${perf.gpu})`);

  // Responsive: no horizontal scroll at phone width.
  await page.setViewport({ width: 390, height: 844, isMobile: true, hasTouch: true });
  await page.waitForFunction(() => window.neurolens?.ready, { timeout: 20000 });
  const overflow = await scene(() => document.documentElement.scrollWidth - window.innerWidth);
  check(overflow <= 0, `no horizontal scroll at 390px (overflow ${overflow}px)`);

  check(errors.length === 0, `no console errors${errors.length ? ': ' + errors.join(' | ') : ''}`);
} finally {
  await browser.close();
  await new Promise((r) => server.httpServer.close(r));
}

console.log(failed ? '\nE2E FAILED' : '\nE2E OK');
process.exit(failed ? 1 : 0);
