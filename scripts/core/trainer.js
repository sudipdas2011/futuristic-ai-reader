import { Tensor } from './tensor.js';

import {
  crossEntropy,
  softmaxCrossEntropyGradient
} from './loss.js';

import { softmax } from './activations.js';

export class Trainer {
  constructor(model, options = {}) {
    this.model = model;

    this.learningRate =
      options.learningRate ?? 0.001;

    this.epochs =
      options.epochs ?? 3;

    this.batchSize =
      options.batchSize ?? 32;

    this.shuffle =
      options.shuffle ?? true;
  }

  train(dataset) {
    if (!dataset || dataset.count === 0) {
      throw new Error(
        'Cannot train on an empty dataset.'
      );
    }

    console.log('');
    console.log('==============================');
    console.log('       MNIST TRAINING');
    console.log('==============================');
    console.log(`Samples: ${dataset.count}`);
    console.log(`Epochs: ${this.epochs}`);
    console.log(`Batch size: ${this.batchSize}`);
    console.log(`Learning rate: ${this.learningRate}`);
    console.log('');

    const history = [];

    for (
      let epoch = 1;
      epoch <= this.epochs;
      epoch++
    ) {
      const indices =
        this.shuffle
          ? this.createShuffledIndices(dataset.count)
          : null;

      let totalLoss = 0;
      let totalCorrect = 0;
      let totalSamples = 0;

      const batchCount =
        Math.ceil(
          dataset.count / this.batchSize
        );

      const epochStart =
        Date.now();

      for (
        let batchIndex = 0;
        batchIndex < batchCount;
        batchIndex++
      ) {
        const start =
          batchIndex * this.batchSize;

        const actualBatchSize =
          Math.min(
            this.batchSize,
            dataset.count - start
          );

        const batch =
          this.createBatch(
            dataset,
            indices,
            start,
            actualBatchSize
          );

        // --------------------------
        // FORWARD
        // --------------------------

        const logits =
          this.model.forward(
            batch.inputs
          );

        const predictions =
          softmax(logits);

        // --------------------------
        // LOSS
        // --------------------------

        const loss =
          crossEntropy(
            predictions,
            batch.targets
          );

        totalLoss +=
          loss * actualBatchSize;

        // --------------------------
        // ACCURACY
        // --------------------------

        const [
          batchSize,
          classCount
        ] = predictions.shape;

        for (
          let b = 0;
          b < batchSize;
          b++
        ) {
          let predictedClass = 0;
          let targetClass = 0;

          for (
            let c = 1;
            c < classCount;
            c++
          ) {
            const index =
              b * classCount + c;

            const predictedCurrent =
              b * classCount +
              predictedClass;

            if (
              predictions.data[index] >
              predictions.data[predictedCurrent]
            ) {
              predictedClass = c;
            }

            const targetIndex =
              b * classCount + c;

            const targetCurrent =
              b * classCount +
              targetClass;

            if (
              batch.targets.data[targetIndex] >
              batch.targets.data[targetCurrent]
            ) {
              targetClass = c;
            }
          }

          if (
            predictedClass === targetClass
          ) {
            totalCorrect++;
          }
        }

        totalSamples +=
          actualBatchSize;

        // --------------------------
        // BACKWARD
        // --------------------------

        const gradient =
          softmaxCrossEntropyGradient(
            predictions,
            batch.targets
          );

        this.model.backward(
          gradient,
          this.learningRate
        );

        // --------------------------
        // PROGRESS
        // --------------------------

        if (
          batchIndex % 10 === 0 ||
          batchIndex === batchCount - 1
        ) {
          const percent =
            (
              ((batchIndex + 1) /
                batchCount) *
              100
            ).toFixed(1);

          process.stdout.write(
            `\rEpoch ${epoch}/${this.epochs} ` +
            `[${percent}%]`
          );
        }
      }

      const averageLoss =
        totalLoss / totalSamples;

      const accuracy =
        totalCorrect / totalSamples;

      const elapsed =
        ((Date.now() - epochStart) / 1000)
          .toFixed(1);

      const result = {
        epoch,
        loss: averageLoss,
        accuracy,
        seconds: Number(elapsed)
      };

      history.push(result);

      console.log('');

      console.log(
        `Epoch ${epoch}/${this.epochs} ` +
        `| loss: ${averageLoss.toFixed(4)} ` +
        `| accuracy: ${(accuracy * 100).toFixed(2)}% ` +
        `| ${elapsed}s`
      );

      console.log('');
    }

    console.log(
      'Training complete.'
    );

    return history;
  }

  createBatch(
    dataset,
    indices,
    start,
    batchSize
  ) {
    const imageSize =
      dataset.rows *
      dataset.cols;

    const inputData =
      new Float32Array(
        batchSize * imageSize
      );

    const targetData =
      new Float32Array(
        batchSize * 10
      );

    for (
      let i = 0;
      i < batchSize;
      i++
    ) {
      const datasetIndex =
        indices
          ? indices[start + i]
          : start + i;

      const imageStart =
        datasetIndex * imageSize;

      const inputStart =
        i * imageSize;

      // MNIST pixels:
      // 0   = background
      // 255 = digit
      for (
        let p = 0;
        p < imageSize;
        p++
      ) {
        inputData[
          inputStart + p
        ] =
          dataset.images[
            imageStart + p
          ] / 255;
      }

      const label =
        dataset.labels[
          datasetIndex
        ];

      targetData[
        i * 10 + label
      ] = 1;
    }

    return {
      inputs: new Tensor(
        inputData,
        [
          batchSize,
          1,
          dataset.rows,
          dataset.cols
        ]
      ),

      targets: new Tensor(
        targetData,
        [
          batchSize,
          10
        ]
      )
    };
  }

  createShuffledIndices(count) {
    const indices =
      new Uint32Array(count);

    for (
      let i = 0;
      i < count;
      i++
    ) {
      indices[i] = i;
    }

    for (
      let i = count - 1;
      i > 0;
      i--
    ) {
      const j =
        Math.floor(
          Math.random() * (i + 1)
        );

      const temp =
        indices[i];

      indices[i] =
        indices[j];

      indices[j] =
        temp;
    }

    return indices;
  }
}