// Which inputs feed a given unit, and how much each one contributes.
// Pure functions over plain arrays (no three.js), so they're testable in Node.

/**
 * Top-k incoming connections of dense unit `j`, ranked by |weight × input|.
 * @returns {{i: number, weight: number, input: number, contribution: number}[]} sorted by |contribution| desc
 */
export function denseTopK({ kernel, kernelShape }, input, j, k) {
  const [nIn, nOut] = kernelShape;
  const idx = new Int32Array(k).fill(-1);
  const mag = new Float32Array(k).fill(-1);
  for (let i = 0; i < nIn; i++) {
    const m = Math.abs(kernel[i * nOut + j] * input[i]);
    if (m <= mag[k - 1]) continue;
    // Insertion into the small sorted top-k buffer.
    let p = k - 1;
    while (p > 0 && mag[p - 1] < m) {
      mag[p] = mag[p - 1];
      idx[p] = idx[p - 1];
      p--;
    }
    mag[p] = m;
    idx[p] = i;
  }
  const out = [];
  for (let p = 0; p < k; p++) {
    const i = idx[p];
    if (i < 0) break;
    const weight = kernel[i * nOut + j];
    out.push({ i, weight, input: input[i], contribution: weight * input[i] });
  }
  return out;
}

/**
 * Receptive field of conv output (r, c, co) with 'same' padding: every in-bounds
 * input cell in the 3x3 window, across all input channels.
 */
export function convInbound({ kernel, kernelShape }, input, r, c, co) {
  const [kh, kw, cin, cout] = kernelShape;
  const [h, w, ch] = input.shape;
  const out = [];
  for (let ky = 0; ky < kh; ky++) {
    for (let kx = 0; kx < kw; kx++) {
      const y = r + ky - (kh >> 1);
      const x = c + kx - (kw >> 1);
      if (y < 0 || x < 0 || y >= h || x >= w) continue;
      for (let ci = 0; ci < cin; ci++) {
        const weight = kernel[((ky * kw + kx) * cin + ci) * cout + co];
        const a = input.data[(y * w + x) * ch + ci];
        out.push({ r: y, c: x, ci, ky, kx, weight, input: a, contribution: weight * a });
      }
    }
  }
  return out;
}

/** The 2x2 window of a max-pool output (r, c, ch) in the (post-ReLU) source map. */
export function poolInbound(source, r, c, ch) {
  const [, w, C] = source.shape;
  const cells = [];
  for (let dy = 0; dy < 2; dy++) {
    for (let dx = 0; dx < 2; dx++) {
      const y = 2 * r + dy;
      const x = 2 * c + dx;
      cells.push({ r: y, c: x, value: source.data[(y * w + x) * C + ch] });
    }
  }
  const max = Math.max(...cells.map((e) => e.value));
  const winner = cells.findIndex((e) => e.value === max);
  return cells.map((e, i) => ({ ...e, isMax: i === winner }));
}

/** Flatten index -> (row, col, channel) of the 7x7x16 pool2 block (Keras NHWC order). */
export function unflatten(i, [, w, ch] = [7, 7, 16]) {
  return { r: Math.floor(i / (w * ch)), c: Math.floor(i / ch) % w, ch: i % ch };
}
