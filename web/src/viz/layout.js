// Scene layout as plain data: where every station, panel and cell sits.
//
// The network runs left -> right along +X. Every station faces the camera (+Z)
// and lies in the z = 0 plane; cells extrude toward +Z by |activation|.
// Feature maps of a stage are stacked in a grid of slices, and the pooled maps
// share their conv maps' rows so each pooled slice sits beside its source.
//
// A "panel" is one 2D grid of cells (a feature map, or one 3x3 kernel slice):
//   cell (r, c) center = (x0 + c * pitch, y0 - r * pitch, 0)
// and `index(r, c)` maps the cell to its element in the layer's flat data.

const STATION_GAP = 26;

export const MAP_PITCH = 1;
export const MAP_CELL = 0.86;

function panel(props) {
  return { pitch: MAP_PITCH, size: MAP_CELL, ...props };
}

/** Lay out `n` tiles in a grid of `cols` columns, returning each tile's top-left corner. */
function tileGrid(n, cols, tileW, tileH, gapX, gapY, left, top) {
  return Array.from({ length: n }, (_, i) => ({
    x: left + (i % cols) * (tileW + gapX),
    y: top - Math.floor(i / cols) * (tileH + gapY),
  }));
}

/** Panel with its top-left corner (not first cell center) at (x, y). */
function panelAt(x, y, props) {
  const p = panel(props);
  return { ...p, x0: x + p.pitch / 2, y0: y - p.pitch / 2 };
}

export function buildLayout() {
  const stations = [];
  let cursor = 0; // left edge of the next station

  function addStation(s) {
    s.left = cursor;
    s.right = cursor + s.width;
    s.center = [cursor + s.width / 2, 0, 0];
    cursor = s.right + (s.gapAfter ?? STATION_GAP);
    stations.push(s);
    return s;
  }

  // 1. Input: 28x28.
  {
    const s = { id: 'input', title: 'Input', width: 28, height: 28 };
    s.layers = {
      input: [panelAt(0, 14, { rows: 28, cols: 28, channel: 0, index: (r, c) => r * 28 + c })],
    };
    shift(addStation(s));
  }

  // 2. Conv1: 8 kernels (3x3x1) beside their 28x28 maps, 4 rows x 2 cols.
  const conv1Rows = { tileH: 28, gapY: 4 };
  {
    const kPitch = 1.4;
    const kW = 3 * kPitch;
    const tileW = kW + 1.5 + 28;
    const tiles = tileGrid(8, 2, tileW, 28, 4, conv1Rows.gapY, 0, 62);
    const s = { id: 'conv1', title: 'Conv1', width: 2 * tileW + 4, height: 124 };
    s.layers = {
      conv1Kernels: tiles.map((t, co) => ({
        ...panelAt(t.x, t.y - 14 + kW / 2, { rows: 3, cols: 3, channel: co, pitch: kPitch, size: 1.25 }),
        index: (r, c) => (r * 3 + c) * 8 + co,
        kernel: true,
      })),
      conv1: tiles.map((t, ch) =>
        panelAt(t.x + kW + 1.5, t.y, { rows: 28, cols: 28, channel: ch, index: (r, c) => (r * 28 + c) * 8 + ch })),
    };
    shift(addStation(s));
  }

  // 3. ReLU1 + Pool1: 8 maps 14x14, aligned with the conv1 rows.
  {
    const tiles = tileGrid(8, 2, 14, 14, 3, 28 + conv1Rows.gapY - 14, 0, 62 - 7);
    const s = { id: 'pool1', title: 'ReLU + Pool1', width: 31, height: 124 };
    s.layers = {
      pool1: tiles.map((t, ch) =>
        panelAt(t.x, t.y, { rows: 14, cols: 14, channel: ch, index: (r, c) => (r * 14 + c) * 8 + ch })),
    };
    shift(addStation(s));
  }

  // 4. Conv2: 16 kernels (3x3x8, shown as 2x4 slices) beside 14x14 maps, 8 rows x 2 cols.
  const conv2Rows = { tileH: 14, gapY: 3, top: 67.5 };
  {
    const kPitch = 0.55;
    const kSlice = 3 * kPitch;
    const kGap = 0.35;
    const kW = 4 * kSlice + 3 * kGap;
    const kH = 2 * kSlice + kGap;
    const tileW = kW + 1.5 + 14;
    const tiles = tileGrid(16, 2, tileW, 14, 4, conv2Rows.gapY, 0, conv2Rows.top);
    const s = { id: 'conv2', title: 'Conv2', width: 2 * tileW + 4, height: 133 };
    const kernels = [];
    tiles.forEach((t, co) => {
      for (let ci = 0; ci < 8; ci++) {
        const kx = t.x + (ci % 4) * (kSlice + kGap);
        const ky = t.y - 7 + kH / 2 - Math.floor(ci / 4) * (kSlice + kGap);
        kernels.push({
          ...panelAt(kx, ky, { rows: 3, cols: 3, channel: co, inChannel: ci, pitch: kPitch, size: 0.48 }),
          index: (r, c) => ((r * 3 + c) * 8 + ci) * 16 + co,
          kernel: true,
        });
      }
    });
    s.layers = {
      conv2Kernels: kernels,
      conv2: tiles.map((t, ch) =>
        panelAt(t.x + kW + 1.5, t.y, { rows: 14, cols: 14, channel: ch, index: (r, c) => (r * 14 + c) * 16 + ch })),
    };
    shift(addStation(s));
  }

  // 5. ReLU2 + Pool2: 16 maps 7x7, aligned with the conv2 rows.
  {
    const tiles = tileGrid(16, 2, 7, 7, 3, 17 - 7, 0, conv2Rows.top - 3.5);
    const s = { id: 'pool2', title: 'ReLU + Pool2', width: 17, height: 133, gapAfter: 42 };
    s.layers = {
      pool2: tiles.map((t, ch) =>
        panelAt(t.x, t.y, { rows: 7, cols: 7, channel: ch, index: (r, c) => (r * 7 + c) * 16 + ch })),
    };
    shift(addStation(s));
  }

  // 6. Flatten: a vertical column of 784 thin bars, in Keras (row, col, channel) order.
  {
    const pitch = 0.17;
    const s = { id: 'flatten', title: 'Flatten', width: 8, height: 784 * pitch, gapAfter: 70 };
    const top = (784 * pitch) / 2 - pitch / 2;
    s.column = { x: 4, top, pitch, barW: 1.6, barH: 0.14 };
    s.block = { x: 4, pitch: 1.1 }; // 7x7x16 block the column unrolls from
    shift(addStation(s));
  }

  // 7. Dense1: 64 neurons in a column.
  {
    const pitch = 2.1;
    const s = { id: 'dense1', title: 'Dense 64 + ReLU', width: 2, height: 64 * pitch, gapAfter: 60 };
    s.neurons = Array.from({ length: 64 }, (_, i) => [1, (63 / 2 - i) * pitch, 0]);
    s.radius = [0.3, 0.95];
    shift(addStation(s));
  }

  // 8. Output: 10 neurons + softmax bars.
  {
    const pitch = 8;
    const s = { id: 'output', title: 'Output · softmax', width: 30, height: 10 * pitch };
    s.neurons = Array.from({ length: 10 }, (_, i) => [2.5, (4.5 - i) * pitch, 0]);
    s.radius = [1.4, 2.6];
    s.bars = { x: 9.5, maxLength: 22, height: 3.2 };
    shift(addStation(s));
  }

  const width = stations.at(-1).right;
  return { stations, byId: Object.fromEntries(stations.map((s) => [s.id, s])), width, center: [width / 2, 0, 0] };
}

/** Move a station's local coordinates (left edge at x = 0) to its world x. */
function shift(s) {
  const dx = s.left;
  for (const panels of Object.values(s.layers ?? {})) {
    for (const p of panels) p.x0 += dx;
  }
  if (s.column) s.column.x += dx;
  if (s.block) s.block.x += dx;
  if (s.neurons) s.neurons = s.neurons.map(([x, y, z]) => [x + dx, y, z]);
  if (s.bars) s.bars.x += dx;
}
