import { Tensor } from './tensor.js';
import { matMul } from './matrix.js';

export class DenseLayer {
  constructor(inputSize, outputSize) {
    this.inputSize = inputSize;
    this.outputSize = outputSize;

    const limit = Math.sqrt(
      6 / (inputSize + outputSize)
    );

    const weights = new Float32Array(
      inputSize * outputSize
    );

    for (let i = 0; i < weights.length; i++) {
      weights[i] =
        (Math.random() * 2 - 1) * limit;
    }

    this.weights = new Tensor(
      weights,
      [inputSize, outputSize]
    );

    this.bias = new Tensor(
      new Float32Array(outputSize),
      [1, outputSize]
    );

    // Saved during forward() for backpropagation.
    this.lastInput = null;
  }

  forward(input) {
    if (input.shape.length !== 2) {
      throw new Error(
        'DenseLayer.forward() expects a 2D tensor.'
      );
    }

    if (input.shape[1] !== this.inputSize) {
      throw new Error(
        `Expected ${this.inputSize} inputs, ` +
        `received ${input.shape[1]}.`
      );
    }

    this.lastInput = input;

    const output = matMul(
      input,
      this.weights
    );

    const [rows, cols] = output.shape;

    const result = new Float32Array(
      output.data
    );

    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        result[row * cols + col] +=
          this.bias.data[col];
      }
    }

    return new Tensor(
      result,
      [rows, cols]
    );
  }

  backward(gradient, learningRate) {
    const input = this.lastInput;

    const [batchSize, inputSize] =
      input.shape;

    const [, outputSize] =
      gradient.shape;

    // dW = input^T × gradient
    const weightGradient =
      new Float32Array(
        inputSize * outputSize
      );

    for (let i = 0; i < inputSize; i++) {
      for (let j = 0; j < outputSize; j++) {
        let sum = 0;

        for (let batch = 0; batch < batchSize; batch++) {
          const inputValue =
            input.data[
              batch * inputSize + i
            ];

          const gradientValue =
            gradient.data[
              batch * outputSize + j
            ];

          sum += inputValue * gradientValue;
        }

        weightGradient[
          i * outputSize + j
        ] = sum / batchSize;
      }
    }

    // db = mean gradient over batch
    const biasGradient =
      new Float32Array(outputSize);

    for (let j = 0; j < outputSize; j++) {
      let sum = 0;

      for (let batch = 0; batch < batchSize; batch++) {
        sum += gradient.data[
          batch * outputSize + j
        ];
      }

      biasGradient[j] =
        sum / batchSize;
    }

    // Gradient flowing backward into previous layer.
    const inputGradient =
      new Float32Array(
        batchSize * inputSize
      );

    for (let batch = 0; batch < batchSize; batch++) {
      for (let i = 0; i < inputSize; i++) {
        let sum = 0;

        for (let j = 0; j < outputSize; j++) {
          sum +=
            gradient.data[
              batch * outputSize + j
            ] *
            this.weights.data[
              i * outputSize + j
            ];
        }

        inputGradient[
          batch * inputSize + i
        ] = sum;
      }
    }

    // Update weights
    for (let i = 0; i < this.weights.size; i++) {
      this.weights.data[i] -=
        learningRate * weightGradient[i];
    }

    // Update biases
    for (let i = 0; i < this.bias.size; i++) {
      this.bias.data[i] -=
        learningRate * biasGradient[i];
    }

    return new Tensor(
      inputGradient,
      [batchSize, inputSize]
    );
  }
}