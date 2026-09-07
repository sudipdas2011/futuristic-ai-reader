import { Tensor } from './tensor.js';

/**
 * Matrix multiplication
 *
 * A: [rowsA, colsA]
 * B: [colsA, colsB]
 *
 * Result: [rowsA, colsB]
 */
export function matMul(A, B) {
  if (A.shape.length !== 2 || B.shape.length !== 2) {
    throw new Error('matMul requires 2D tensors.');
  }

  const [rowsA, colsA] = A.shape;
  const [rowsB, colsB] = B.shape;

  if (colsA !== rowsB) {
    throw new Error(
      `Cannot multiply [${rowsA},${colsA}] × [${rowsB},${colsB}]. ` +
      `Inner dimensions must match.`
    );
  }

  const result = new Float32Array(rowsA * colsB);

  for (let row = 0; row < rowsA; row++) {
    for (let col = 0; col < colsB; col++) {
      let sum = 0;

      for (let k = 0; k < colsA; k++) {
        const a = A.data[row * colsA + k];
        const b = B.data[k * colsB + col];

        sum += a * b;
      }

      result[row * colsB + col] = sum;
    }
  }

  return new Tensor(result, [rowsA, colsB]);
}