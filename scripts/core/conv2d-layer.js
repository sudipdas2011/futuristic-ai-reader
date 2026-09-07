import { Tensor } from './tensor.js';

export class Conv2D {
  constructor(
    inputHeight,
    inputWidth,
    kernelSize,
    filterCount
  ) {
    this.inputHeight = inputHeight;
    this.inputWidth = inputWidth;
    this.kernelSize = kernelSize;
    this.filterCount = filterCount;

    this.outputHeight =
      inputHeight - kernelSize + 1;

    this.outputWidth =
      inputWidth - kernelSize + 1;

    const limit = Math.sqrt(
      6 / (kernelSize * kernelSize + 1)
    );

    this.kernels = [];

    for (let filter = 0; filter < filterCount; filter++) {
      const values = new Float32Array(
        kernelSize * kernelSize
      );

      for (let i = 0; i < values.length; i++) {
        values[i] =
          (Math.random() * 2 - 1) * limit;
      }

      this.kernels.push(
        new Tensor(
          values,
          [kernelSize, kernelSize]
        )
      );
    }

    this.biases = new Float32Array(filterCount);

    this.lastInput = null;
  }

  forward(input) {
    if (
      input.shape[0] !== this.inputHeight ||
      input.shape[1] !== this.inputWidth
    ) {
      throw new Error(
        `Expected input [${this.inputHeight},${this.inputWidth}], ` +
        `received [${input.shape.join(',')}].`
      );
    }

    this.lastInput = input;

    const output = new Float32Array(
      this.filterCount *
      this.outputHeight *
      this.outputWidth
    );

    for (let filter = 0; filter < this.filterCount; filter++) {
      const kernel = this.kernels[filter];

      for (let y = 0; y < this.outputHeight; y++) {
        for (let x = 0; x < this.outputWidth; x++) {
          let sum = 0;

          for (let ky = 0; ky < this.kernelSize; ky++) {
            for (let kx = 0; kx < this.kernelSize; kx++) {
              const inputValue =
                input.data[
                  (y + ky) * this.inputWidth +
                  (x + kx)
                ];

              const kernelValue =
                kernel.data[
                  ky * this.kernelSize + kx
                ];

              sum += inputValue * kernelValue;
            }
          }

          const index =
            filter *
              this.outputHeight *
              this.outputWidth +
            y * this.outputWidth +
            x;

          output[index] =
            sum + this.biases[filter];
        }
      }
    }

    return new Tensor(
      output,
      [
        this.filterCount,
        this.outputHeight,
        this.outputWidth
      ]
    );
  }

  backward(gradient, learningRate) {
    if (!this.lastInput) {
      throw new Error(
        'Conv2D.backward() called before forward().'
      );
    }

    const input = this.lastInput;

    const inputGradient =
      new Float32Array(input.size);

    const kernelGradients =
      this.kernels.map(
        kernel =>
          new Float32Array(kernel.size)
      );

    const biasGradients =
      new Float32Array(this.filterCount);

    for (
      let filter = 0;
      filter < this.filterCount;
      filter++
    ) {
      for (
        let y = 0;
        y < this.outputHeight;
        y++
      ) {
        for (
          let x = 0;
          x < this.outputWidth;
          x++
        ) {
          const gradientIndex =
            filter *
              this.outputHeight *
              this.outputWidth +
            y * this.outputWidth +
            x;

          const grad =
            gradient.data[gradientIndex];

          biasGradients[filter] += grad;

          for (
            let ky = 0;
            ky < this.kernelSize;
            ky++
          ) {
            for (
              let kx = 0;
              kx < this.kernelSize;
              kx++
            ) {
              const inputIndex =
                (y + ky) * this.inputWidth +
                (x + kx);

              const kernelIndex =
                ky * this.kernelSize + kx;

              kernelGradients[filter][kernelIndex] +=
                input.data[inputIndex] * grad;

              inputGradient[inputIndex] +=
                this.kernels[filter].data[kernelIndex] *
                grad;
            }
          }
        }
      }
    }

    for (
      let filter = 0;
      filter < this.filterCount;
      filter++
    ) {
      const kernel = this.kernels[filter];

      for (let i = 0; i < kernel.size; i++) {
        kernel.data[i] -=
          learningRate *
          kernelGradients[filter][i];
      }

      this.biases[filter] -=
        learningRate *
        biasGradients[filter];
    }

    return new Tensor(
      inputGradient,
      [...input.shape]
    );
  }
}