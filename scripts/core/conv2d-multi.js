import { Tensor } from './tensor.js';

export class Conv2DMulti {
  constructor(
    inputChannels,
    inputHeight,
    inputWidth,
    kernelSize,
    filterCount
  ) {
    this.inputChannels = inputChannels;
    this.inputHeight = inputHeight;
    this.inputWidth = inputWidth;
    this.kernelSize = kernelSize;
    this.filterCount = filterCount;

    this.outputHeight =
      inputHeight - kernelSize + 1;

    this.outputWidth =
      inputWidth - kernelSize + 1;

    if (this.outputHeight <= 0 || this.outputWidth <= 0) {
      throw new Error(
        'Kernel is too large for the input.'
      );
    }

    /*
      Each filter contains one kernel
      for every input channel.

      filter 0:
        channel 0 → 3×3
        channel 1 → 3×3
        channel 2 → 3×3
        ...

      All of them are summed into
      ONE output feature map.
    */

    const limit = Math.sqrt(
      6 /
        (
          inputChannels *
          kernelSize *
          kernelSize +
          filterCount
        )
    );

    this.kernels = [];

    for (
      let filter = 0;
      filter < filterCount;
      filter++
    ) {
      const values = new Float32Array(
        inputChannels *
        kernelSize *
        kernelSize
      );

      for (let i = 0; i < values.length; i++) {
        values[i] =
          (Math.random() * 2 - 1) * limit;
      }

      this.kernels.push(
        new Tensor(
          values,
          [
            inputChannels,
            kernelSize,
            kernelSize
          ]
        )
      );
    }

    this.biases =
      new Float32Array(filterCount);

    this.lastInput = null;
  }

  forward(input) {
    if (input.shape.length !== 3) {
      throw new Error(
        'Conv2DMulti.forward() expects [channels, height, width].'
      );
    }

    const [
      channels,
      height,
      width
    ] = input.shape;

    if (
      channels !== this.inputChannels ||
      height !== this.inputHeight ||
      width !== this.inputWidth
    ) {
      throw new Error(
        `Expected [${this.inputChannels},${this.inputHeight},${this.inputWidth}], ` +
        `received [${input.shape.join(',')}].`
      );
    }

    this.lastInput = input;

    const output = new Float32Array(
      this.filterCount *
      this.outputHeight *
      this.outputWidth
    );

    for (
      let filter = 0;
      filter < this.filterCount;
      filter++
    ) {
      const kernel =
        this.kernels[filter];

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
          let sum = 0;

          for (
            let channel = 0;
            channel < this.inputChannels;
            channel++
          ) {
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
                  channel *
                    this.inputHeight *
                    this.inputWidth +
                  (y + ky) *
                    this.inputWidth +
                  (x + kx);

                const kernelIndex =
                  channel *
                    this.kernelSize *
                    this.kernelSize +
                  ky *
                    this.kernelSize +
                  kx;

                sum +=
                  input.data[inputIndex] *
                  kernel.data[kernelIndex];
              }
            }
          }

          const outputIndex =
            filter *
              this.outputHeight *
              this.outputWidth +
            y *
              this.outputWidth +
            x;

          output[outputIndex] =
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

  backward(
    gradient,
    learningRate
  ) {
    if (!this.lastInput) {
      throw new Error(
        'Conv2DMulti.backward() called before forward().'
      );
    }

    const input =
      this.lastInput;

    const inputGradient =
      new Float32Array(input.size);

    const kernelGradients =
      this.kernels.map(
        kernel =>
          new Float32Array(
            kernel.size
          )
      );

    const biasGradients =
      new Float32Array(
        this.filterCount
      );

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
            y *
              this.outputWidth +
            x;

          const grad =
            gradient.data[
              gradientIndex
            ];

          biasGradients[filter] += grad;

          for (
            let channel = 0;
            channel < this.inputChannels;
            channel++
          ) {
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
                  channel *
                    this.inputHeight *
                    this.inputWidth +
                  (y + ky) *
                    this.inputWidth +
                  (x + kx);

                const kernelIndex =
                  channel *
                    this.kernelSize *
                    this.kernelSize +
                  ky *
                    this.kernelSize +
                  kx;

                kernelGradients[
                  filter
                ][kernelIndex] +=
                  input.data[inputIndex] *
                  grad;

                inputGradient[inputIndex] +=
                  this.kernels[
                    filter
                  ].data[kernelIndex] *
                  grad;
              }
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
      const kernel =
        this.kernels[filter];

      for (
        let i = 0;
        i < kernel.size;
        i++
      ) {
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