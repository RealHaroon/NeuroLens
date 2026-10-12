// The subset of TensorFlow.js this app needs: core ops, the Layers API and the
// CPU backend. Importing the full @tensorflow/tfjs bundle would also pull in the
// WebGL backend, the graph-model converter and tfjs-data (~3x the size).
import '@tensorflow/tfjs-backend-cpu';

export * from '@tensorflow/tfjs-core';
export * from '@tensorflow/tfjs-layers';
