// Instanced building blocks. Every grid cell and neuron is one instance of a
// shared geometry: one draw call per layer, and a new input only rewrites the
// instance color/matrix arrays (no scene rebuild, no allocation).

import {
  BoxGeometry, BufferAttribute, BufferGeometry, DynamicDrawUsage, InstancedBufferAttribute,
  InstancedMesh, LineBasicMaterial, LineSegments, MeshLambertMaterial, SphereGeometry,
} from 'three';
import { LUTS, lutIndex } from './colors.js';

export const BOX = new BoxGeometry(1, 1, 1);
const SPHERE = new SphereGeometry(1, 18, 12);

export function instanced(geometry, count) {
  const mesh = new InstancedMesh(geometry, new MeshLambertMaterial({ color: 0xffffff }), count);
  mesh.instanceMatrix.setUsage(DynamicDrawUsage);
  mesh.instanceColor = new InstancedBufferAttribute(new Float32Array(count * 3), 3);
  mesh.instanceColor.setUsage(DynamicDrawUsage);
  mesh.frustumCulled = false; // instances move; a stale bounding sphere would cull them
  return mesh;
}

/** Write a scale+translate matrix straight into an instance matrix array. */
export function writeMatrix(arr, i, x, y, z, sx, sy, sz) {
  const o = i * 16;
  arr.fill(0, o, o + 16);
  arr[o] = sx;
  arr[o + 5] = sy;
  arr[o + 10] = sz;
  arr[o + 12] = x;
  arr[o + 13] = y;
  arr[o + 14] = z;
  arr[o + 15] = 1;
}

function writeColor(arr, i, lut, slot) {
  arr[i * 3] = lut[slot * 3];
  arr[i * 3 + 1] = lut[slot * 3 + 1];
  arr[i * 3 + 2] = lut[slot * 3 + 2];
}

/**
 * A set of 2D panels (feature maps or kernel slices) drawn as extruded boxes.
 * Cells extrude toward +Z by |value| / max, so a tilted view reads as a relief.
 */
export class GridLayer {
  constructor(name, panels, { colormap, depth = [0.15, 2.2] }) {
    this.name = name;
    this.panels = panels;
    this.lut = LUTS[colormap];
    this.signed = colormap === 'diverging' || colormap === 'kernel';
    this.depthRange = depth;

    this.offsets = [];
    let count = 0;
    for (const p of panels) {
      this.offsets.push(count);
      count += p.rows * p.cols;
    }
    this.count = count;
    this.depth = new Float32Array(count);
    this.values = new Float32Array(count);
    this.mesh = instanced(BOX, count);
    this.mesh.name = name;
  }

  /**
   * Recolor and re-extrude every cell from a layer's flat data array.
   * `reveal(x, y)` (optional, 0..1) flattens and fades cells toward zero, for build-up animations.
   */
  update(data, max, reveal) {
    const mat = this.mesh.instanceMatrix.array;
    const col = this.mesh.instanceColor.array;
    const [d0, d1] = this.depthRange;
    let i = 0;
    for (const p of this.panels) {
      for (let r = 0; r < p.rows; r++) {
        for (let c = 0; c < p.cols; c++, i++) {
          const v = data ? data[p.index(r, c)] : 0;
          const n = max > 0 ? Math.min(1, Math.abs(v) / max) : 0;
          const x = p.x0 + c * p.pitch;
          const y = p.y0 - r * p.pitch;
          let d = d0 + n * d1;
          let slot = lutIndex(v, max, this.signed);
          if (reveal) {
            const t = reveal(x, y);
            const zero = this.signed ? 128 : 0;
            d = Math.max(0.02, d * t);
            slot = Math.round(zero + (slot - zero) * t);
          }
          this.values[i] = v;
          this.depth[i] = d;
          writeColor(col, i, this.lut, slot);
          writeMatrix(mat, i, x, y, d / 2, p.size, p.size, d);
        }
      }
    }
    this.mesh.instanceMatrix.needsUpdate = true;
    this.mesh.instanceColor.needsUpdate = true;
  }

  /** Instance under world point (x, y), or null. */
  pick(x, y) {
    for (let k = 0; k < this.panels.length; k++) {
      const p = this.panels[k];
      const c = Math.round((x - p.x0) / p.pitch);
      const r = Math.round((p.y0 - y) / p.pitch);
      if (r < 0 || c < 0 || r >= p.rows || c >= p.cols) continue;
      // Only count hits on the cell itself, not the gap around it.
      if (Math.abs(x - (p.x0 + c * p.pitch)) > p.pitch / 2 || Math.abs(p.y0 - r * p.pitch - y) > p.pitch / 2) continue;
      return this.cell(k, r, c);
    }
    return null;
  }

  cell(panelIndex, r, c) {
    const p = this.panels[panelIndex];
    const i = this.offsets[panelIndex] + r * p.cols + c;
    return {
      layer: this.name, panel: p, panelIndex, r, c, instance: i,
      value: this.values[i],
      center: [p.x0 + c * p.pitch, p.y0 - r * p.pitch, this.depth[i] / 2],
      size: [p.size, p.size, this.depth[i]],
    };
  }
}

/** Neurons as instanced spheres; radius and brightness follow the activation. */
export class NeuronLayer {
  // colorFloor: lowest LUT slot used, so inactive neurons stay visible against the background.
  constructor(name, positions, radius, { colormap = 'sequential', colorFloor = 0 } = {}) {
    this.colorFloor = colorFloor;
    this.name = name;
    this.positions = positions;
    this.radius = radius;
    this.lut = LUTS[colormap];
    this.count = positions.length;
    this.r = new Float32Array(this.count);
    this.values = new Float32Array(this.count);
    this.mesh = instanced(SPHERE, this.count);
    this.mesh.name = name;
  }

  /** `scale(i)` (optional) multiplies a neuron's radius and brightness, for swell/bounce animations. */
  update(data, max, scale) {
    const mat = this.mesh.instanceMatrix.array;
    const col = this.mesh.instanceColor.array;
    const [r0, r1] = this.radius;
    for (let i = 0; i < this.count; i++) {
      const v = data ? data[i] : 0;
      const n = max > 0 ? Math.min(1, Math.max(0, v) / max) : 0;
      const k = scale ? Math.max(0.02, scale(i)) : 1;
      const r = (r0 + n * (r1 - r0)) * k;
      const [x, y, z] = this.positions[i];
      this.values[i] = v;
      this.r[i] = r;
      writeColor(col, i, this.lut, Math.round((this.colorFloor + n * (255 - this.colorFloor)) * Math.min(1, k)));
      writeMatrix(mat, i, x, y, z, r, r, r);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
    this.mesh.instanceColor.needsUpdate = true;
  }

  pick(x, y) {
    for (let i = 0; i < this.count; i++) {
      const [px, py] = this.positions[i];
      const hit = Math.max(this.r[i], 0.9);
      if ((x - px) ** 2 + (y - py) ** 2 <= hit * hit) return this.cell(i);
    }
    return null;
  }

  cell(i) {
    const r = this.r[i];
    return { layer: this.name, index: i, instance: i, value: this.values[i], center: this.positions[i], size: [2 * r, 2 * r, 2 * r], radius: r };
  }
}

/** A growable-up-to-`max` batch of colored line segments with per-vertex alpha. */
export class LineBatch {
  constructor(max, { opacity = 1 } = {}) {
    this.max = max;
    this.n = 0;
    this.positions = new Float32Array(max * 6);
    this.colors = new Float32Array(max * 8);
    const geom = new BufferGeometry();
    geom.setAttribute('position', new BufferAttribute(this.positions, 3).setUsage(DynamicDrawUsage));
    geom.setAttribute('color', new BufferAttribute(this.colors, 4).setUsage(DynamicDrawUsage));
    geom.setDrawRange(0, 0);
    this.material = new LineBasicMaterial({ vertexColors: true, transparent: true, opacity, depthWrite: false });
    this.lines = new LineSegments(geom, this.material);
    this.lines.frustumCulled = false;
  }

  clear() {
    this.n = 0;
  }

  push(a, b, color, alpha) {
    if (this.n >= this.max) return;
    const i = this.n++;
    this.positions.set(a, i * 6);
    this.positions.set(b, i * 6 + 3);
    this.colors.set([color.r, color.g, color.b, alpha, color.r, color.g, color.b, alpha], i * 8);
  }

  commit() {
    const geom = this.lines.geometry;
    geom.attributes.position.needsUpdate = true;
    geom.attributes.color.needsUpdate = true;
    geom.setDrawRange(0, this.n * 2);
  }
}
