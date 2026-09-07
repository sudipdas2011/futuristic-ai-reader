import { Tensor } from '../core/tensor.js';

import { Conv2DBatch } from '../core/conv2d-batch.js';

import { MaxPool2D } from '../core/maxpool2d.js';

import {
  relu,
  reluBackward,
  softmax
} from '../core/activations.js';

import {
  flattenBatch
} from '../core/flatten.js';

import { DenseLayer } from '../core/dense.js';

export class CNNReaderV3 {
  constructor(
    imageHeight,
    imageWidth,
    classCount
  ) {
    this.imageHeight = imageHeight;
    this.imageWidth = imageWidth;
    this.classCount = classCount;

    // --------------------------------
    // CONVOLUTION BLOCK 1
    // --------------------------------

    this.conv1 =
      new Conv2DBatch(
        1,              // input channels
        imageHeight,
        imageWidth,
        3,              // kernel
        8               // filters
      );

    this.pool1 =
      new MaxPool2D(2, 2);

    // --------------------------------
    // CONVOLUTION BLOCK 2
    // --------------------------------

    const conv1Height =
      imageHeight - 3 + 1;

    const conv1Width =
      imageWidth - 3 + 1;

    const pool1Height =
      Math.floor(
        (conv1Height - 2) / 2
      ) + 1;

    const pool1Width =
      Math.floor(
        (conv1Width - 2) / 2
      ) + 1;

    this.conv2 =
      new Conv2DBatch(
        8,
        pool1Height,
        pool1Width,
        3,
        16
      );

    this.pool2 =
      new MaxPool2D(2, 2);

    // --------------------------------
    // CALCULATE DENSE INPUT
    // --------------------------------

    const conv2Height =
      pool1Height - 3 + 1;

    const conv2Width =
      pool1Width - 3 + 1;

    const pool2Height =
      Math.floor(
        (conv2Height - 2) / 2
      ) + 1;

    const pool2Width =
      Math.floor(
        (conv2Width - 2) / 2
      ) + 1;

    this.featureSize =
      16 *
      pool2Height *
      pool2Width;

    // --------------------------------
    // OUTPUT CLASSIFIER
    // --------------------------------

    this.dense =
      new DenseLayer(
        this.featureSize,
        classCount
      );

    // --------------------------------
    // CACHE
    // --------------------------------

    this.lastConv1 =
      null;

    this.lastRelu1 =
      null;

    this.lastPool1 =
      null;

    this.lastConv2 =
      null;

    this.lastRelu2 =
      null;

    this.lastPool2 =
      null;
  }

  forward(input) {
    if (input.shape.length !== 4) {
      throw new Error(
        'CNNReaderV3.forward() expects [batch, channels, height, width].'
      );
    }

    if (input.shape[1] !== 1) {
      throw new Error(
        'CNNReaderV3 currently expects 1 input channel.'
      );
    }

    // --------------------------------
    // BLOCK 1
    // --------------------------------

    this.lastConv1 =
      this.conv1.forward(input);

    this.lastRelu1 =
      relu(this.lastConv1);

    this.lastPool1 =
      this.pool1.forward(
        this.lastRelu1
      );

    // --------------------------------
    // BLOCK 2
    // --------------------------------

    this.lastConv2 =
      this.conv2.forward(
        this.lastPool1
      );

    this.lastRelu2 =
      relu(this.lastConv2);

    this.lastPool2 =
      this.pool2.forward(
        this.lastRelu2
      );

    // --------------------------------
    // CLASSIFIER
    // --------------------------------

    const flattened =
      flattenBatch(
        this.lastPool2
      );

    return this.dense.forward(
      flattened
    );
  }

  predict(input) {
    const logits =
      this.forward(input);

    return softmax(logits);
  }

  backward(
    gradient,
    learningRate
  ) {
    // --------------------------------
    // DENSE
    // --------------------------------

    const denseGradient =
      this.dense.backward(
        gradient,
        learningRate
      );

    // --------------------------------
    // RESTORE CNN SHAPE
    // --------------------------------

    const pool2Gradient =
      new Tensor(
        new Float32Array(
          denseGradient.data
        ),
        this.lastPool2.shape
      );

    // --------------------------------
    // POOL 2
    // --------------------------------

    const relu2Gradient =
      this.pool2.backward(
        pool2Gradient
      );

    // --------------------------------
    // RELU 2
    // --------------------------------

    const conv2Gradient =
      reluBackward(
        relu2Gradient,
        this.lastConv2
      );

    // --------------------------------
    // CONV 2
    // --------------------------------

    const pool1Gradient =
      this.conv2.backward(
        conv2Gradient,
        learningRate
      );

    // --------------------------------
    // POOL 1
    // --------------------------------

    const relu1Gradient =
      this.pool1.backward(
        pool1Gradient
      );

    // --------------------------------
    // RELU 1
    // --------------------------------

    const conv1Gradient =
      reluBackward(
        relu1Gradient,
        this.lastConv1
      );

    // --------------------------------
    // CONV 1
    // --------------------------------

    this.conv1.backward(
      conv1Gradient,
      learningRate
    );
  }
}