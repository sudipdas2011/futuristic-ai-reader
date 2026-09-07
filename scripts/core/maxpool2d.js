import { Tensor } from './tensor.js';

export class MaxPool2D {
  constructor(poolSize = 2, stride = 2) {
    this.poolSize = poolSize;
    this.stride = stride;

    this.lastInput = null;
    this.maxIndices = null;
    this.outputShape = null;
  }

  forward(input) {
    if (input.shape.length !== 4) {
      throw new Error(
        'MaxPool2D.forward() expects [batch, channels, height, width].'
      );
    }

    const [
      batchSize,
      channels,
      inputHeight,
      inputWidth
    ] = input.shape;

    const outputHeight =
      Math.floor(
        (inputHeight - this.poolSize) /
          this.stride
      ) + 1;

    const outputWidth =
      Math.floor(
        (inputWidth - this.poolSize) /
          this.stride
      ) + 1;

    if (
      outputHeight <= 0 ||
      outputWidth <= 0
    ) {
      throw new Error(
        'Pooling window is too large for the input.'
      );
    }

    this.lastInput = input;

    this.outputShape = [
      batchSize,
      channels,
      outputHeight,
      outputWidth
    ];

    const outputSize =
      batchSize *
      channels *
      outputHeight *
      outputWidth;

    const output =
      new Float32Array(outputSize);

    // For every output value, remember
    // which input pixel produced the maximum.
    this.maxIndices =
      new Int32Array(outputSize);

    const inputImageSize =
      channels *
      inputHeight *
      inputWidth;

    const outputImageSize =
      channels *
      outputHeight *
      outputWidth;

    for (
      let batch = 0;
      batch < batchSize;
      batch++
    ) {
      for (
        let channel = 0;
        channel < channels;
        channel++
      ) {
        for (
          let y = 0;
          y < outputHeight;
          y++
        ) {
          for (
            let x = 0;
            x < outputWidth;
            x++
          ) {
            let max = -Infinity;
            let maxIndex = -1;

            for (
              let py = 0;
              py < this.poolSize;
              py++
            ) {
              for (
                let px = 0;
                px < this.poolSize;
                px++
              ) {
                const inputY =
                  y * this.stride + py;

                const inputX =
                  x * this.stride + px;

                const index =
                  batch *
                    inputImageSize +
                  channel *
                    inputHeight *
                    inputWidth +
                  inputY *
                    inputWidth +
                  inputX;

                if (
                  input.data[index] > max
                ) {
                  max =
                    input.data[index];

                  maxIndex = index;
                }
              }
            }

            const outputIndex =
              batch *
                outputImageSize +
              channel *
                outputHeight *
                outputWidth +
              y *
                outputWidth +
              x;

            output[outputIndex] = max;
            this.maxIndices[outputIndex] =
              maxIndex;
          }
        }
      }
    }

    return new Tensor(
      output,
      this.outputShape
    );
  }

  backward(gradient) {
    if (
      !this.lastInput ||
      !this.maxIndices
    ) {
      throw new Error(
        'MaxPool2D.backward() called before forward().'
      );
    }

    if (
      gradient.size !==
      this.maxIndices.length
    ) {
      throw new Error(
        'Gradient size does not match pooling output.'
      );
    }

    const inputGradient =
      new Float32Array(
        this.lastInput.size
      );

    for (
      let i = 0;
      i < gradient.size;
      i++
    ) {
      const inputIndex =
        this.maxIndices[i];

      inputGradient[inputIndex] +=
        gradient.data[i];
    }

    return new Tensor(
      inputGradient,
      [...this.lastInput.shape]
    );
  }
}