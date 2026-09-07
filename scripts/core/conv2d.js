import { Tensor } from './tensor.js';

export function conv2d(input, kernel, bias = 0, stride = 1) {
  if (input.shape.length !== 2) {
    throw new Error('conv2d input must be a 2D tensor.');
  }

  if (kernel.shape.length !== 2) {
    throw new Error('conv2d kernel must be a 2D tensor.');
  }

  const [inputHeight, inputWidth] = input.shape;
  const [kernelHeight, kernelWidth] = kernel.shape;

  const outputHeight =
    Math.floor(
      (inputHeight - kernelHeight) / stride
    ) + 1;

  const outputWidth =
    Math.floor(
      (inputWidth - kernelWidth) / stride
    ) + 1;

  if (outputHeight <= 0 || outputWidth <= 0) {
    throw new Error(
      `Kernel [${kernelHeight},${kernelWidth}] is too large ` +
      `for input [${inputHeight},${inputWidth}].`
    );
  }

  const output = new Float32Array(
    outputHeight * outputWidth
  );

  for (let outY = 0; outY < outputHeight; outY++) {
    for (let outX = 0; outX < outputWidth; outX++) {
      let sum = 0;

      const startY = outY * stride;
      const startX = outX * stride;

      for (let ky = 0; ky < kernelHeight; ky++) {
        for (let kx = 0; kx < kernelWidth; kx++) {
          const inputY = startY + ky;
          const inputX = startX + kx;

          const inputValue =
            input.data[
              inputY * inputWidth + inputX
            ];

          const kernelValue =
            kernel.data[
              ky * kernelWidth + kx
            ];

          sum += inputValue * kernelValue;
        }
      }

      output[
        outY * outputWidth + outX
      ] = sum + bias;
    }
  }

  return new Tensor(
    output,
    [outputHeight, outputWidth]
  );
}