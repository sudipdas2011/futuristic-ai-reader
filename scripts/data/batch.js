import { Tensor } from '../core/tensor.js';

export function createBatch(samples) {
  if (samples.length === 0) {
    throw new Error(
      'Cannot create a batch from zero samples.'
    );
  }

  const inputSize =
    samples[0].input.size;

  const targetSize =
    samples[0].target.size;

  const inputs =
    new Float32Array(
      samples.length * inputSize
    );

  const targets =
    new Float32Array(
      samples.length * targetSize
    );

  for (
    let sample = 0;
    sample < samples.length;
    sample++
  ) {
    const input =
      samples[sample].input;

    const target =
      samples[sample].target;

    inputs.set(
      input.data,
      sample * inputSize
    );

    targets.set(
      target.data,
      sample * targetSize
    );
  }

  return {
    inputs: new Tensor(
      inputs,
      [
        samples.length,
        inputSize
      ]
    ),

    targets: new Tensor(
      targets,
      [
        samples.length,
        targetSize
      ]
    )
  };
}