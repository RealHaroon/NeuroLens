// Colormaps as 256-entry lookup tables in linear RGB (three.js' working space),
// so updating thousands of instance colors is a straight array copy.
//
// The 3D viewport is a dark display surface in both page themes, so:
//  - sequential (magnitude): one blue hue, near-zero recedes into the background
//  - diverging (signed, conv outputs): blue <- dark gray -> red, zero recedes
//  - kernel weights: blue <- white -> red (white = zero weight, as in most CNN figures)
//  - input pixels: grayscale, brightness = pixel value

import { Color, SRGBColorSpace } from 'three';

export const SCENE_BG = '#0e1117';

const STOPS = {
  sequential: ['#1b2230', '#184f95', '#3987e5', '#9ec5f4', '#f2f7ff'],
  grayscale: ['#1b2230', '#ffffff'],
  // Diverging maps are sampled over [-1, 1]; the middle stop is zero.
  diverging: ['#3987e5', '#1f5aa8', '#2b3242', '#b13f2c', '#e5634a'],
  kernel: ['#2a6fd6', '#9ec5f4', '#f4f4f2', '#f2a08e', '#d33a2a'],
};

function buildLut(stops) {
  const colors = stops.map((hex) => new Color(hex));
  const lut = new Float32Array(256 * 3);
  const c = new Color();
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (stops.length - 1);
    const k = Math.min(stops.length - 2, Math.floor(t));
    c.copy(colors[k]).lerp(colors[k + 1], t - k);
    lut.set([c.r, c.g, c.b], i * 3);
  }
  return lut;
}

export const LUTS = Object.fromEntries(Object.entries(STOPS).map(([k, v]) => [k, buildLut(v)]));
export const SIGNED = new Set(['diverging', 'kernel']);

/** LUT slot for a value: sequential maps [0, max] -> 0..255, signed maps [-max, max] -> 0..255. */
export function lutIndex(value, max, signed) {
  const t = max > 0 ? value / max : 0;
  const u = signed ? (t + 1) / 2 : t;
  return Math.max(0, Math.min(255, Math.round(u * 255)));
}

/** CSS gradient for a legend swatch. */
export function cssGradient(name) {
  return `linear-gradient(90deg, ${STOPS[name].join(', ')})`;
}

export const SIGN_COLORS = {
  positive: new Color().setStyle('#e5634a', SRGBColorSpace),
  negative: new Color().setStyle('#3987e5', SRGBColorSpace),
};
export const ACCENT = new Color('#9ec5f4');
export const MUTED = new Color('#3a4252');
