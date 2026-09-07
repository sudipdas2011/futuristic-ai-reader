import { Tensor } from './tensor.js';

/**
 * Cross-entropy loss for one-hot targets.
 *
 * predictions: [batch, classes]
 * targets:     [batch, classes]
 */
export function crossEntropy(predictions, targets) {
  if (
    predictions.shape.length !== 2 ||
    targets.shape.length !== 2
  ) {
    throw new Error(
      'crossEntropy expects 2D tensors.'
    );
  }

  if (
    predictions.shape[0] !== targets.shape[0] ||
    predictions.shape[1] !== targets.shape[1]
  ) {
    throw new Error(
      'Predictions and targets must have the same shape.'
    );
  }

  const [batch, classes] = predictions.shape;
  let total = 0;

  for (let row = 0; row < batch; row++) {
    for (let col = 0; col < classes; col++) {
      const target = targets.data[row * classes + col];

      if (target === 0) continue;

      // Prevent log(0)
      const probability = Math.max(
        predictions.data[row * classes + col],
        1e-7
      );

      total -= target * Math.log(probability);
    }
  }

  return total / batch;
}

/**
 * Gradient of softmax + cross entropy.
 *
 * For:
 *
 *   loss = crossEntropy(softmax(logits), target)
 *
 * the gradient simplifies to:
 *
 *   prediction - target
 */
export function softmaxCrossEntropyGradient(
  predictions,
  targets
) {
  if (predictions.size !== targets.size) {
    throw new Error(
      'Predictions and targets must have the same size.'
    );
  }

  const gradient = new Float32Array(
    predictions.size
  );

  for (let i = 0; i < predictions.size; i++) {
    gradient[i] =
      predictions.data[i] - targets.data[i];
  }

  return new Tensor(
    gradient,
    [...predictions.shape]
  );
}