// Defines, compiles and prepares to train a TensorFlow.js model to recognize handwritten digits
// NODE_NO_WARNINGS=1 node example-1.js

// Note: This project uses several deprecated npm packages. Key warnings include:
// - `inflight@1.0.6`: Leaks memory; consider using `lru-cache` for async request handling.
// - `npmlog@5.0.1`, `are-we-there-yet@2.0.0`, and `gauge@3.0.2`: No longer supported.
// - `rimraf@2.7.1` and `rimraf@3.0.2`: Versions prior to v4 are unsupported.
// - `glob@7.2.3`: Versions prior to v9 are unsupported.
//
// Consider updating dependencies to supported versions or alternatives where possible.

const tf = require('@tensorflow/tfjs-node');
// tf.env().set('DEBUG', true);

const util_1 = {
  isNullOrUndefined: (value) => value === null || value === undefined,
};

// Define the model architecture
const model = tf.sequential();
model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [784] }));
model.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

// Compile the model
model.compile({ loss: 'categoricalCrossentropy', optimizer: 'adam', metrics: ['accuracy'] });

// Mock getNextBatch function
function getNextBatch() {
  const xs = tf.randomUniform([32, 784]); // Ensure this always returns valid tensors
  const ys = tf.oneHot(tf.cast(tf.randomUniform([32], 0, 10, 'int32'), 'int32'), 10); // Ensure valid labels
  return { xs, ys };
}

// Prepare the data
const data = tf.data.generator(function* () {
  while (true) {
    const { xs, ys } = getNextBatch();
    yield { xs, ys };
  }
}).batch(32);

(async () => {
  console.log('Training started...');
  try {
    await model.fitDataset(data, {
      epochs: 1,
      batchesPerEpoch: 10,
    });
    console.log('Training completed.');
  } catch (err) {
    console.error('Training error:', err);
  }
})();
