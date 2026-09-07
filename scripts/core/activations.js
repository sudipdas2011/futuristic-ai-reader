import { Tensor } from './tensor.js';

export function relu(tensor) {
  return tensor.map(value => Math.max(0, value));
}

export function reluBackward(gradient, input) {
  if (gradient.size !== input.size) {
    throw new Error(
      'reluBackward requires gradient and input to have the same size.'
    );
  }

  const result = new Float32Array(input.size);

  for (let i = 0; i < input.size; i++) {
    result[i] =
      input.data[i] > 0
        ? gradient.data[i]
        : 0;
  }

  return new Tensor(result, [...input.shape]);
}

export function softmax(tensor) {
  if (tensor.shape.length !== 2) {
    throw new Error('softmax expects a 2D tensor.');
  }

  const [rows, cols] = tensor.shape;
  const result = new Float32Array(tensor.size);

  for (let row = 0; row < rows; row++) {
    const offset = row * cols;

    let max = -Infinity;

    for (let col = 0; col < cols; col++) {
      max = Math.max(
        max,
        tensor.data[offset + col]
      );
    }

    let sum = 0;

    for (let col = 0; col < cols; col++) {
      const value = Math.exp(
        tensor.data[offset + col] - max
      );

      result[offset + col] = value;
      sum += value;
    }

    for (let col = 0; col < cols; col++) {
      result[offset + col] /= sum;
    }
  }

  return new Tensor(
    result,
    [...tensor.shape]
  );
}