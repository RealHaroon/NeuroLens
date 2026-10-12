// End-to-end check of the production build in headless Chrome:
// draws digits with the mouse, uploads a dark-on-white image, and verifies the
// predictions, the per-layer activations, and that tensors don't leak.
//
// Usage:  npm run test:e2e          (CHROME_PATH overrides the browser binary)
//         SCREENSHOT_DIR=... saves screenshots of the light and dark themes.

import { mkdtempSync, writeFileSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build, preview } from 'vite';
import puppeteer from 'puppeteer-core';

const CHROME = process.env.CHROME_PATH ?? '/usr/bin/google-chrome';
const SHOTS = process.env.SCREENSHOT_DIR;
const root = fileURLToPath(new URL('..', import.meta.url));

await build({ root, logLevel: 'warn' });
const server = await preview({ root, preview: { port: 4174, strictPort: true }, logLevel: 'warn' });
const url = server.resolvedUrls.local[0];

const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, args: ['--no-sandbox'] });
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
  await page.waitForFunction((prev) => window.neurolens.last?.softmax.data[7] > 0.5, { timeout: 5000 }).catch(() => {});
  const up = await page.evaluate(() => {
    const p = Array.from(window.neurolens.last.softmax.data);
    return { pred: p.indexOf(Math.max(...p)), conf: Math.max(...p) };
  });
  console.log('Upload:');
  check(up.pred === parity.labels[0], `dark-on-white upload -> predicted ${up.pred} (${(up.conf * 100).toFixed(1)}%), label ${parity.labels[0]}`);

  if (SHOTS) {
    await draw(cases[1][1]);
    await page.screenshot({ path: join(SHOTS, 'phase2-light.png') });
    await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: 'dark' }]);
    await page.screenshot({ path: join(SHOTS, 'phase2-dark.png') });
    await page.setViewport({ width: 390, height: 844, isMobile: true });
    await page.screenshot({ path: join(SHOTS, 'phase2-mobile.png'), fullPage: true });
    console.log(`Screenshots saved to ${SHOTS}`);
  }

  check(errors.length === 0, `no console errors${errors.length ? ': ' + errors.join(' | ') : ''}`);
} finally {
  await browser.close();
  await new Promise((r) => server.httpServer.close(r));
}

console.log(failed ? '\nE2E FAILED' : '\nE2E OK');
process.exit(failed ? 1 : 0);
