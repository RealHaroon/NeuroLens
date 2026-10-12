// Plain-English notes for each station of the forward pass, plus a live fact
// computed from the current activations.

const pct = (x) => `${(x * 100).toFixed(0)}%`;
const zeros = (a) => a.reduce((n, v) => n + (v === 0 ? 1 : 0), 0) / a.length;
const argmax = (a) => a.reduce((b, v, i) => (v > a[b] ? i : b), 0);

export const OVERVIEW = {
  title: 'The whole network',
  shape: '28×28×1 → 10',
  layers: ['conv1', 'conv2', 'dense1', 'dense2'],
  body: [
    'A small convolutional neural network (CNN) that reads a 28×28 grayscale image and outputs ten probabilities, one per digit.',
    'Two convolution blocks find visual patterns (strokes, then curves and corners). Two dense layers combine them into a decision.',
    'Press <b>Play forward pass</b> to follow one image through every layer, or click a layer above.',
  ],
  live: (r) => `Prediction: <b>${argmax(r.softmax.data)}</b> (${(Math.max(...r.softmax.data) * 100).toFixed(1)}%)`,
};

export const STEPS = {
  input: {
    title: 'Input',
    shape: '28×28×1',
    layers: [],
    body: [
      'The preprocessed drawing: 784 numbers between 0 (background) and 1 (ink), exactly what the network receives.',
      'Like the MNIST training data, the digit is scaled into a 20×20 box and centered by its center of mass.',
    ],
    live: (r) => `${pct(1 - zeros(r.input.data))} of pixels contain ink`,
  },
  conv1: {
    title: 'Convolution 1',
    shape: '28×28×8',
    layers: ['conv1'],
    body: [
      'Each of the 8 <b>filters</b> is a 3×3 grid of learned weights (the small squares). It slides across the image. At every position it multiplies the 9 pixels under it by its weights, adds them up, and adds a bias.',
      'The result is a <b>feature map</b>: red where the filter’s pattern (say, a horizontal edge) is present, blue where the opposite appears.',
      '“Same” padding keeps the output 28×28. Turn on <b>Kernel slide</b> to watch one map being built.',
    ],
    live: (r) => `${pct(r.conv1.data.filter((v) => v < 0).length / r.conv1.data.length)} of outputs are negative (blue)`,
  },
  pool1: {
    title: 'ReLU + max-pool 1',
    shape: '14×14×8',
    layers: [],
    body: [
      '<b>ReLU</b> replaces every negative value with 0, so the blue disappears and only “pattern found” evidence remains.',
      '<b>Max-pooling</b> keeps the largest value in each 2×2 block. That halves the width and height while keeping the strongest responses, and it makes the network less sensitive to small shifts.',
    ],
    live: (r) => `ReLU zeroed ${pct(zeros(r.relu1.data))} of conv1 outputs`,
  },
  conv2: {
    title: 'Convolution 2',
    shape: '14×14×16',
    layers: ['conv2'],
    body: [
      'The same idea one level up. Each of the 16 filters looks at a 3×3 patch across <b>all 8</b> maps from the previous layer (72 weights, drawn as 8 small slices).',
      'Because its inputs are already edge detectors, it can combine them into larger parts of digits: corners, loops, curves.',
    ],
    live: (r) => `${pct(r.conv2.data.filter((v) => v < 0).length / r.conv2.data.length)} of outputs are negative (blue)`,
  },
  pool2: {
    title: 'ReLU + max-pool 2',
    shape: '7×7×16',
    layers: [],
    body: [
      'ReLU and 2×2 max-pooling again. What is left is a coarse 7×7 grid where each of the 16 channels says how strongly its pattern appears in each region of the digit.',
    ],
    live: (r) => `ReLU zeroed ${pct(zeros(r.relu2.data))} of conv2 outputs`,
  },
  flatten: {
    title: 'Flatten',
    shape: '784',
    layers: [],
    body: [
      'Unrolls the 7×7×16 block into one list of 784 numbers (row by row, all 16 channels of a position together) so a dense layer can look at everything at once.',
      'Nothing is learned here; it only reshapes. Press <b>Unroll flatten</b> to see it.',
    ],
    live: (r) => `${pct(1 - zeros(r.flatten.data))} of the 784 values are non-zero`,
  },
  dense1: {
    title: 'Dense 1 + ReLU',
    shape: '64',
    layers: ['dense1'],
    body: [
      'Each of the 64 neurons computes a weighted sum of all 784 numbers plus a bias, then applies ReLU. Bigger, brighter spheres are more active.',
      'Only each neuron’s strongest inputs (by |weight × input|) are drawn. Red adds to the sum, blue subtracts. This layer holds most of the model’s parameters.',
    ],
    live: (r) => `${r.relu3.data.filter((v) => v > 0).length} of 64 neurons are active`,
  },
  output: {
    title: 'Output + softmax',
    shape: '10',
    layers: ['dense2'],
    body: [
      'Ten neurons, one per digit, each a weighted sum of the 64 dense neurons. These raw scores are called <b>logits</b>.',
      '<b>Softmax</b> turns the logits into probabilities that add up to 100%. The largest one is the prediction.',
    ],
    live: (r) => {
      const p = r.softmax.data;
      const order = Array.from(p.keys()).sort((a, b) => p[b] - p[a]);
      return `Prediction <b>${order[0]}</b> (${(p[order[0]] * 100).toFixed(1)}%), runner-up ${order[1]} (${(p[order[1]] * 100).toFixed(1)}%)`;
    },
  },
};

/** Trainable parameters for a step, from the exported model spec. */
export function paramsFor(step, spec) {
  let total = 0;
  const parts = [];
  for (const name of step.layers) {
    const l = spec.layers.find((x) => x.name === name);
    const n = Object.values(l.weights).reduce((a, w) => a + w.shape.reduce((x, y) => x * y, 1), 0);
    total += n;
    const k = l.weights.kernel.shape.join('×');
    parts.push(`${name}: ${k} weights + ${l.weights.bias.shape[0]} biases`);
  }
  return { total, parts };
}
