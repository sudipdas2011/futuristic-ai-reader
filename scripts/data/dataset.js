import { Tensor } from '../core/tensor.js';

export class Dataset {
  constructor(samples = []) {
    this.samples = samples;
  }

  add(input, target) {
    if (!(input instanceof Tensor)) {
      throw new TypeError(
        'Dataset input must be a Tensor.'
      );
    }

    if (!(target instanceof Tensor)) {
      throw new TypeError(
        'Dataset target must be a Tensor.'
      );
    }

    this.samples.push({
      input,
      target
    });
  }

  get size() {
    return this.samples.length;
  }

  shuffle() {
    for (
      let i = this.samples.length - 1;
      i > 0;
      i--
    ) {
      const j =
        Math.floor(Math.random() * (i + 1));

      [
        this.samples[i],
        this.samples[j]
      ] = [
        this.samples[j],
        this.samples[i]
      ];
    }

    return this;
  }

  batch(batchSize) {
    const batches = [];

    for (
      let start = 0;
      start < this.samples.length;
      start += batchSize
    ) {
      const group =
        this.samples.slice(
          start,
          start + batchSize
        );

      batches.push(group);
    }

    return batches;
  }
}