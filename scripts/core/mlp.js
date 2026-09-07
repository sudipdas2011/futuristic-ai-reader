import { DenseLayer } from './dense.js';
import {
  relu,
  reluBackward,
  softmax
} from './activations.js';

export class MLP {
  constructor(inputSize, hiddenSize, outputSize) {
    this.hidden = new DenseLayer(
      inputSize,
      hiddenSize
    );

    this.output = new DenseLayer(
      hiddenSize,
      outputSize
    );

    this.lastHiddenInput = null;
  }

  forward(input) {
    this.lastHiddenInput =
      this.hidden.forward(input);

    const activated =
      relu(this.lastHiddenInput);

    return this.output.forward(
      activated
    );
  }

  predict(input) {
    return softmax(
      this.forward(input)
    );
  }

  backward(gradient, learningRate) {
    const hiddenGradient =
      this.output.backward(
        gradient,
        learningRate
      );

    const reluGradient =
      reluBackward(
        hiddenGradient,
        this.lastHiddenInput
      );

    this.hidden.backward(
      reluGradient,
      learningRate
    );
  }
}