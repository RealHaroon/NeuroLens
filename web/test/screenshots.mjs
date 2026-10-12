// Regenerates the README screenshots in docs/screenshots/ from the production
// build, driving headless Chrome on the real GPU.
//
// Usage:  npm run screenshots

import { mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build, preview } from 'vite';
import puppeteer from 'puppeteer-core';

const root = fileURLToPath(new URL('..', import.meta.url));
const OUT = fileURLToPath(new URL('../../docs/screenshots/', import.meta.url));
mkdirSync(OUT, { recursive: true });

await build({ root, logLevel: 'warn' });
const server = await preview({ root, preview: { port: 4177, strictPort: true }, logLevel: 'warn' });
const browser = await puppeteer.launch({
  executablePath: process.env.CHROME_PATH ?? '/usr/bin/google-chrome',
  headless: true,
  args: ['--use-angle=gl', '--enable-unsafe-swiftshader'],
});

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function open(width, height, scheme = 'light') {
  const page = await browser.newPage();
  await page.setViewport({ width, height, deviceScaleFactor: 1 });
  await page.emulateMediaFeatures([{ name: 'prefers-color-scheme', value: scheme }]);
  await page.goto(server.resolvedUrls.local[0]);
  await page.waitForFunction(() => window.neurolens?.scene, { timeout: 30000 });
  return page;
}

/** Draw a "3" with two mouse strokes and return the prediction. */
async function drawThree(page) {
  const box = await (await page.$('#drawpad')).boundingBox();
  const s = box.width / 280;
  const arc = (cx, cy, rx, ry, a0, a1, n = 18) =>
    Array.from({ length: n + 1 }, (_, i) => {
      const t = a0 + ((a1 - a0) * i) / n;
      return [box.x + (cx + rx * Math.cos(t)) * s, box.y + (cy + ry * Math.sin(t)) * s];
    });
  const path = [...arc(140, 85, 55, 45, -Math.PI * 0.9, Math.PI * 0.5), ...arc(140, 185, 60, 52, -Math.PI * 0.5, Math.PI * 0.9)];
  await page.mouse.move(...path[0]);
  await page.mouse.down();
  for (const p of path.slice(1)) await page.mouse.move(...p, { steps: 2 });
  await page.mouse.up();
  await sleep(300);
  return page.evaluate(() => {
    const p = Array.from(window.neurolens.last.softmax.data);
    return p.indexOf(Math.max(...p));
  });
}

const scene = (page, fn, ...args) => page.evaluate(fn, ...args);
const frame = (page) => page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));

try {
  const page = await open(1600, 960);
  console.log(`Drew a digit, predicted ${await drawThree(page)}`);
  await frame(page);
  await page.screenshot({ path: join(OUT, 'overview.png') });

  await scene(page, () => window.neurolens.scene.focus('conv1', 0));
  await frame(page);
  let p = await scene(page, () => window.neurolens.scene.debug.screenOf('conv1', 1, 9, 13));
  await page.mouse.move(p.x, p.y);
  await frame(page);
  await page.screenshot({ path: join(OUT, 'conv1-hover.png') });

  await scene(page, () => window.neurolens.scene.focus('dense1', 0));
  await frame(page);
  const strongest = await scene(page, () => {
    const a = Array.from(window.neurolens.last.relu3.data);
    return a.indexOf(Math.max(...a));
  });
  p = await scene(page, (j) => window.neurolens.scene.debug.screenOf('dense1', j), strongest);
  await page.mouse.move(p.x, p.y);
  await frame(page);
  await page.screenshot({ path: join(OUT, 'dense-topk.png') });
  await page.mouse.move(5, 5);

  // Playback mid-way: layers ahead are still dark.
  await scene(page, () => window.neurolens.player.step(1)); // input
  await scene(page, () => window.neurolens.player.step(1)); // conv1
  await scene(page, () => window.neurolens.player.step(1)); // pool1
  await sleep(600);
  await scene(page, () => window.neurolens.scene.focusRange(['input', 'conv2'], 0));
  await sleep(300);
  await page.screenshot({ path: join(OUT, 'playback.png') });

  await page.click('#kernel-slide');
  await scene(page, () => window.neurolens.player.goto('input'));
  await page.click('#step-forward');
  await sleep(1100 + 3500);
  await page.screenshot({ path: join(OUT, 'kernel-slide.png') });
  await scene(page, () => window.neurolens.scene.cancelAnimations());

  await page.click('#unroll');
  await sleep(900 + 1500);
  await page.screenshot({ path: join(OUT, 'flatten-unroll.png') });
  await page.close();

  const dark = await open(1600, 960, 'dark');
  await drawThree(dark);
  await scene(dark, () => window.neurolens.player.goto('output'));
  await scene(dark, () => window.neurolens.scene.focus('output', 0));
  await frame(dark);
  await dark.screenshot({ path: join(OUT, 'dark-output.png') });
  await dark.close();

  const phone = await browser.newPage();
  await phone.setViewport({ width: 390, height: 844, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  await phone.goto(server.resolvedUrls.local[0]);
  await phone.waitForFunction(() => window.neurolens?.scene, { timeout: 30000 });
  await drawThree(phone);
  await phone.screenshot({ path: join(OUT, 'mobile.png'), fullPage: true });
  console.log(`Screenshots written to ${OUT}`);
} finally {
  await browser.close();
  await new Promise((r) => server.httpServer.close(r));
}
