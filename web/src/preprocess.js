// MNIST-style preprocessing: turn an arbitrary drawing or photo of a digit into
// the 28x28 [0,1] image the model was trained on.
//
// MNIST digits are white-on-black, fitted into a 20x20 box (aspect preserved)
// and placed in a 28x28 frame so their center of mass sits at the center.
//
// The core works on plain grayscale arrays ({data: Float32Array, width, height},
// values in [0,1]) so it can be tested in Node; canvasToGray() is the browser glue.

const SIZE = 28;
const BOX = 20;
const INK_THRESHOLD = 0.1;

/** Read a canvas (or anything drawable) into a grayscale array, compositing alpha onto white. */
export function canvasToGray(source) {
  const width = source.width;
  const height = source.height;
  const ctx = source.getContext?.('2d', { willReadFrequently: true });
  let rgba;
  if (ctx) {
    rgba = ctx.getImageData(0, 0, width, height).data;
  } else {
    const tmp = document.createElement('canvas');
    tmp.width = width;
    tmp.height = height;
    const tctx = tmp.getContext('2d', { willReadFrequently: true });
    tctx.drawImage(source, 0, 0);
    rgba = tctx.getImageData(0, 0, width, height).data;
  }
  const data = new Float32Array(width * height);
  for (let i = 0; i < data.length; i++) {
    const a = rgba[i * 4 + 3] / 255;
    const luma = (0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2]) / 255;
    data[i] = luma * a + (1 - a); // transparent pixels count as white paper
  }
  return { data, width, height };
}

/**
 * Full pipeline. Returns { image: Float32Array(784), empty: boolean, inverted: boolean }.
 * `empty` is true when no ink was found (the image is then all zeros).
 */
export function preprocess(gray) {
  const inverted = shouldInvert(gray);
  let img = inverted ? invert(gray) : gray;
  img = stretchContrast(img);

  const box = boundingBox(img, INK_THRESHOLD);
  if (!box) return { image: new Float32Array(SIZE * SIZE), empty: true, inverted };

  const cropped = crop(img, box);
  const scale = BOX / Math.max(cropped.width, cropped.height);
  const w = Math.max(1, Math.round(cropped.width * scale));
  const h = Math.max(1, Math.round(cropped.height * scale));
  const small = resizeArea(cropped, w, h);

  // Place so the center of mass lands at the frame center.
  const [cx, cy] = centerOfMass(small);
  const ox = clamp(Math.round(SIZE / 2 - cx), 0, SIZE - w);
  const oy = clamp(Math.round(SIZE / 2 - cy), 0, SIZE - h);
  const image = new Float32Array(SIZE * SIZE);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      image[(y + oy) * SIZE + (x + ox)] = small.data[y * w + x];
    }
  }
  return { image, empty: false, inverted };
}

/** Background is judged from the border: a bright border means dark ink on white paper. */
export function shouldInvert({ data, width, height }) {
  let sum = 0;
  let n = 0;
  for (let x = 0; x < width; x++) {
    sum += data[x] + data[(height - 1) * width + x];
    n += 2;
  }
  for (let y = 1; y < height - 1; y++) {
    sum += data[y * width] + data[y * width + width - 1];
    n += 2;
  }
  return sum / n > 0.5;
}

function invert({ data, width, height }) {
  return { data: data.map((v) => 1 - v), width, height };
}

/** Map the background level to 0 and the brightest ink to 1 (helps faint photos). */
function stretchContrast({ data, width, height }) {
  // Median via a 256-bin histogram (a full sort is slow on 280x280 at 60 Hz).
  const hist = new Uint32Array(256);
  let hi = 0;
  for (const v of data) {
    hist[Math.min(255, Math.max(0, Math.round(v * 255)))]++;
    if (v > hi) hi = v;
  }
  let lo = 0;
  for (let i = 0, seen = 0; i < 256; i++) {
    seen += hist[i];
    if (seen > data.length / 2) {
      lo = i / 255; // median ≈ background
      break;
    }
  }
  if (hi - lo < 1e-3) return { data: new Float32Array(data.length), width, height };
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = clamp((data[i] - lo) / (hi - lo), 0, 1);
  return { data: out, width, height };
}

function boundingBox({ data, width, height }, threshold) {
  let x0 = width;
  let y0 = height;
  let x1 = -1;
  let y1 = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (data[y * width + x] > threshold) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  return x1 < 0 ? null : { x: x0, y: y0, width: x1 - x0 + 1, height: y1 - y0 + 1 };
}

function crop({ data, width }, box) {
  const out = new Float32Array(box.width * box.height);
  for (let y = 0; y < box.height; y++) {
    const row = (box.y + y) * width + box.x;
    out.set(data.subarray(row, row + box.width), y * box.width);
  }
  return { data: out, width: box.width, height: box.height };
}

/** Area-averaging (box filter) resize: each output pixel is the mean of the source area it covers. */
export function resizeArea(src, dw, dh) {
  const rows = resample1d(src.data, src.width, src.height, dw, true);
  const out = resample1d(rows, dw, src.height, dh, false);
  return { data: out, width: dw, height: dh };
}

// Resample along x (horizontal=true) or y with exact fractional coverage weights.
function resample1d(data, w, h, n, horizontal) {
  const len = horizontal ? w : h;
  const other = horizontal ? h : w;
  const out = new Float32Array(horizontal ? n * h : w * n);
  const ratio = len / n;
  for (let i = 0; i < n; i++) {
    const start = i * ratio;
    const end = start + ratio;
    for (let o = 0; o < other; o++) {
      let acc = 0;
      for (let s = Math.floor(start); s < Math.min(Math.ceil(end), len); s++) {
        const cover = Math.min(end, s + 1) - Math.max(start, s);
        acc += cover * (horizontal ? data[o * w + s] : data[s * w + o]);
      }
      out[horizontal ? o * n + i : i * w + o] = acc / ratio;
    }
  }
  return out;
}

function centerOfMass({ data, width, height }) {
  let m = 0;
  let mx = 0;
  let my = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const v = data[y * width + x];
      m += v;
      mx += v * (x + 0.5);
      my += v * (y + 0.5);
    }
  }
  return m > 0 ? [mx / m, my / m] : [width / 2, height / 2];
}

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}
