import { Tensor } from '../core/tensor.js';
import { Conv2D } from '../core/conv2d-layer.js';
import { DenseLayer } from '../core/dense.js';
import { relu, reluBackward, softmax } from '../core/activations.js';
import { flatten } from '../core/flatten.js';

export class CNNReader {
  constructor(
    imageHeight,
    imageWidth,
    classCount
  ) {
    this.imageHeight = imageHeight;
    this.imageWidth = imageWidth;
    this.classCount = classCount;

    // Image
    //   ↓
    // Conv 3×3, 8 filters
    //   ↓
    // ReLU
    //   ↓
    // Flatten
    //   ↓
    // Dense
    //   ↓
    // Softmax

    this.conv = new Conv2D(
      imageHeight,
      imageWidth,
      3,
      8
    );

    this.convOutputSize =
      8 *
      (imageHeight - 2) *
      (imageWidth - 2);

    this.dense = new DenseLayer(
      this.convOutputSize,
      classCount
    );

    this.lastConvInput = null;
    this.lastConvOutput = null;
    this.lastActivated = null;
  }

  forward(input) {
    this.lastConvInput = input;

    const convOutput =
      this.conv.forward(input);

    this.lastConvOutput =
      convOutput;

    const activated =
      relu(convOutput);

    this.lastActivated =
      activated;

    const flattened =
      flatten(activated);

    return this.dense.forward(
      flattened
    );
  }

  predict(input) {
    const logits =
      this.forward(input);

    return softmax(logits);
  }

  backward(gradient, learningRate) {
    // Dense backward expects [batch, features].
    const denseGradient =
      this.dense.backward(
        gradient,
        learningRate
      );

    // Rebuild the convolution output shape.
    const convGradient =
      new Tensor(
        new Float32Array(
          denseGradient.data
        ),
        this.lastConvOutput.shape
      );

    const activatedGradient =
      reluBackward(
        convGradient,
        this.lastConvOutput
      );

    this.conv.backward(
      activatedGradient,
      learningRate
    );
  }
}