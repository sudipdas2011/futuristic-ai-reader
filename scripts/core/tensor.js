// scripts/core/tensor.js

export class Tensor {
  constructor(data, shape = null) {
    if (data instanceof Float32Array) {
      this.data = data;
    } else {
      this.data = new Float32Array(data);
    }

    this.shape = shape ?? [this.data.length];

    const expectedSize = this.shape.reduce(
      (total, value) => total * value,
      1
    );

    if (expectedSize !== this.data.length) {
      throw new Error(
        `Shape ${this.shape.join('×')} requires ${expectedSize} values, ` +
        `but received ${this.data.length}.`
      );
    }
  }

  get size() {
    return this.data.length;
  }

  get(index) {
    return this.data[index];
  }

  set(index, value) {
    this.data[index] = value;
  }

  map(fn) {
    const result = new Float32Array(this.size);

    for (let i = 0; i < this.size; i++) {
      result[i] = fn(this.data[i], i);
    }

    return new Tensor(result, [...this.shape]);
  }

  add(other) {
    this.#checkSameSize(other);

    return this.map((value, i) => value + other.data[i]);
  }

  subtract(other) {
    this.#checkSameSize(other);

    return this.map((value, i) => value - other.data[i]);
  }

  multiply(value) {
    if (value instanceof Tensor) {
      this.#checkSameSize(value);

      return this.map((x, i) => x * value.data[i]);
    }

    return this.map(x => x * value);
  }

  sum() {
    let total = 0;

    for (const value of this.data) {
      total += value;
    }

    return total;
  }

  max() {
    let result = -Infinity;

    for (const value of this.data) {
      if (value > result) {
        result = value;
      }
    }

    return result;
  }

  argMax() {
    let index = 0;

    for (let i = 1; i < this.size; i++) {
      if (this.data[i] > this.data[index]) {
        index = i;
      }
    }

    return index;
  }

  clone() {
    return new Tensor(
      new Float32Array(this.data),
      [...this.shape]
    );
  }

  toArray() {
    return Array.from(this.data);
  }

  toString() {
    return `Tensor(${this.shape.join('×')}): [${this.toArray().join(', ')}]`;
  }

  #checkSameSize(other) {
    if (!(other instanceof Tensor)) {
      throw new TypeError('Expected another Tensor.');
    }

    if (this.size !== other.size) {
      throw new Error(
        `Tensor size mismatch: ${this.size} vs ${other.size}`
      );
    }
  }
}