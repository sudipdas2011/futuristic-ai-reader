import { Tensor } from '../core/tensor.js';
import { Conv2DMulti } from '../core/conv2d-multi.js';
import { MaxPool2D } from '../core/maxpool2d.js';
import { DenseLayer } from '../core/dense.js';
import {
  relu,
  reluBackward,
  softmax
} from '../core/activations.js';
import { flatten } from '../core/flatten.js';

export class CNNReaderV2 {
  constructor(imageHeight, imageWidth, classCount) {
    this.imageHeight = imageHeight;
    this.imageWidth = imageWidth;
    this.classCount = classCount;

    // ─────────────────────────────
    // Block 1
    // ─────────────────────────────

    this.conv1 = new Conv2DMulti(
      1,
      imageHeight,
      imageWidth,
      3,
      8
    );

    this.pool1 = new MaxPool2D(2, 2);

    const conv1Height = imageHeight - 2;
    const conv1Width = imageWidth - 2;

    const pool1Height =
      Math.floor((conv1Height - 2) / 2) + 1;

    const pool1Width =
      Math.floor((conv1Width - 2) / 2) + 1;

    // ─────────────────────────────
    // Block 2
    // ─────────────────────────────

    this.conv2 = new Conv2DMulti(
      8,
      pool1Height,
      pool1Width,
      3,
      16
    );

    this.pool2 = new MaxPool2D(2, 2);

    const conv2Height =
      pool1Height - 2;

    const conv2Width =
      pool1Width - 2;

    const pool2Height =
      Math.floor((conv2Height - 2) / 2) + 1;

    const pool2Width =
      Math.floor((conv2Width - 2) / 2) + 1;

    this.flattenSize =
      16 *
      pool2Height *
      pool2Width;

    // ─────────────────────────────
    // Classifier
    // ─────────────────────────────

    this.dense = new DenseLayer(
      this.flattenSize,
      classCount
    );

    // Cached tensors for backward pass.
    this.conv1Output = null;
    this.relu1Output = null;
    this.pool1Output = null;

    this.conv2Output = null;
    this.relu2Output = null;
    this.pool2Output = null;
  }

  forward(input) {
    // Conv 1
    this.conv1Output =
      this.conv1.forward(input);

    // ReLU 1
    this.relu1Output =
      relu(this.conv1Output);

    // Pool 1
    this.pool1Output =
      this.pool1.forward(
        this.relu1Output
      );

    // Conv 2
    this.conv2Output =
      this.conv2.forward(
        this.pool1Output
      );

    // ReLU 2
    this.relu2Output =
      relu(this.conv2Output);

    // Pool 2
    this.pool2Output =
      this.pool2.forward(
        this.relu2Output
      );

    // Flatten
    const flattened =
      flatten(this.pool2Output);

    // Classifier
    return this.dense.forward(
      flattened
    );
  }

  predict(input) {
    return softmax(
      this.forward(input)
    );
  }

  backward(gradient, learningRate) {
    // Dense
    const denseGradient =
      this.dense.backward(
        gradient,
        learningRate
      );

    // Reshape gradient back to pool2
    const pool2Gradient =
      new Tensor(
        new Float32Array(
          denseGradient.data
        ),
        this.pool2Output.shape
      );

    // Pool 2
    const relu2Gradient =
      this.pool2.backward(
        pool2Gradient
      );

    // ReLU 2
    const conv2Gradient =
      reluBackward(
        relu2Gradient,
        this.conv2Output
      );

    // Conv 2
    const pool1Gradient =
      this.conv2.backward(
        conv2Gradient,
        learningRate
      );

    // Pool 1
    const relu1Gradient =
      this.pool1.backward(
        pool1Gradient
      );

    // ReLU 1
    const conv1Gradient =
      reluBackward(
        relu1Gradient,
        this.conv1Output
      );

    // Conv 1
    this.conv1.backward(
      conv1Gradient,
      learningRate
    );
  }
}