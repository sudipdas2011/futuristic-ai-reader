import { Tensor } from './tensor.js';

export function flatten(tensor) {
  if (tensor.shape.length === 0) {
    throw new Error(
      'flatten() requires a tensor with at least one dimension.'
    );
  }

  return new Tensor(
    new Float32Array(tensor.data),
    [1, tensor.size]
  );
}

export function flattenBatch(tensor) {
  if (tensor.shape.length !== 4) {
    throw new Error(
      'flattenBatch() expects [batch, channels, height, width].'
    );
  }

  const [
    batchSize,
    channels,
    height,
    width
  ] = tensor.shape;

  const features =
    channels *
    height *
    width;

  return new Tensor(
    new Float32Array(tensor.data),
    [batchSize, features]
  );
}