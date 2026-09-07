import { Tensor } from "../core/tensor.js";
import {
  getImage,
  getLabel,
} from "./mnist.js";

export function createBatch(
  dataset,
  start,
  batchSize
) {
  const actualSize = Math.min(
    batchSize,
    dataset.count - start
  );

  const imageSize =
    dataset.rows * dataset.cols;

  const inputData = new Float32Array(
    actualSize * imageSize
  );

  const targetData = new Float32Array(
    actualSize * 10
  );

  for (let i = 0; i < actualSize; i++) {
    const index = start + i;

    const image = getImage(
      dataset,
      index
    );

    const label = getLabel(
      dataset,
      index
    );

    const inputOffset =
      i * imageSize;

    // MNIST:
    // background = 0
    // digit      = 255
    for (let p = 0; p < imageSize; p++) {
      inputData[inputOffset + p] =
        image[p] / 255;
    }

    // One-hot target
    targetData[i * 10 + label] = 1;
  }

  return {
    inputs: new Tensor(
      inputData,
      [
        actualSize,
        1,
        dataset.rows,
        dataset.cols,
      ]
    ),

    targets: new Tensor(
      targetData,
      [
        actualSize,
        10,
      ]
    ),

    size: actualSize,
  };
}

export function shuffleIndices(count) {
  const indices = new Uint32Array(count);

  for (let i = 0; i < count; i++) {
    indices[i] = i;
  }

  for (
    let i = count - 1;
    i > 0;
    i--
  ) {
    const j =
      Math.floor(
        Math.random() * (i + 1)
      );

    const temp = indices[i];
    indices[i] = indices[j];
    indices[j] = temp;
  }

  return indices;
}

export function createShuffledBatch(
  dataset,
  indices,
  start,
  batchSize
) {
  const actualSize = Math.min(
    batchSize,
    dataset.count - start
  );

  const imageSize =
    dataset.rows * dataset.cols;

  const inputData = new Float32Array(
    actualSize * imageSize
  );

  const targetData = new Float32Array(
    actualSize * 10
  );

  for (let i = 0; i < actualSize; i++) {
    const datasetIndex =
      indices[start + i];

    const image = getImage(
      dataset,
      datasetIndex
    );

    const label = getLabel(
      dataset,
      datasetIndex
    );

    const inputOffset =
      i * imageSize;

    for (let p = 0; p < imageSize; p++) {
      inputData[inputOffset + p] =
        image[p] / 255;
    }

    targetData[
      i * 10 + label
    ] = 1;
  }

  return {
    inputs: new Tensor(
      inputData,
      [
        actualSize,
        1,
        dataset.rows,
        dataset.cols,
      ]
    ),

    targets: new Tensor(
      targetData,
      [
        actualSize,
      10,
    ]
    ),

    size: actualSize,
  };
}