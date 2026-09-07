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
      options.epochs ?? 10;

    this.batchSize =
      options.batchSize ?? 32;

    this.shuffle =
      options.shuffle ?? true;
  }

  train(dataset) {
    if (dataset.size === 0) {
      throw new Error(
        'Cannot train on an empty dataset.'
      );
    }

    console.log(
      `Training ${dataset.size} samples`
    );

    console.log(
      `Epochs: ${this.epochs}`
    );

    console.log(
      `Batch size: ${this.batchSize}`
    );

    console.log(
      `Learning rate: ${this.learningRate}`
    );

    console.log('');

    const history = [];

    for (
      let epoch = 1;
      epoch <= this.epochs;
      epoch++
    ) {
      if (this.shuffle) {
        dataset.shuffle();
      }

      let totalLoss = 0;
      let totalCorrect = 0;
      let totalSamples = 0;

      const batches =
        dataset.batch(
          this.batchSize
        );

      for (
        const samples of batches
      ) {
        const {
          inputs,
          targets
        } = this.createImageBatch(
          samples
        );

        // --------------------------
        // ONE FORWARD PASS
        // --------------------------

        const logits =
          this.model.forward(
            inputs
          );

        // Convert logits to
        // probabilities.
        const predictions =
          softmax(logits);

        // --------------------------
        // LOSS
        // --------------------------

        const loss =
          crossEntropy(
            predictions,
            targets
          );

        totalLoss +=
          loss * samples.length;

        // --------------------------
        // ACCURACY
        // --------------------------

        const [
          batchSize,
          classCount
        ] = predictions.shape;

        for (
          let batch = 0;
          batch < batchSize;
          batch++
        ) {
          let predictionClass = 0;
          let targetClass = 0;

          for (
            let classIndex = 1;
            classIndex < classCount;
            classIndex++
          ) {
            const predictionIndex =
              batch * classCount +
              classIndex;

            const currentPrediction =
              batch * classCount +
              predictionClass;

            if (
              predictions.data[
                predictionIndex
              ] >
              predictions.data[
                currentPrediction
              ]
            ) {
              predictionClass =
                classIndex;
            }

            const targetIndex =
              batch * classCount +
              classIndex;

            const currentTarget =
              batch * classCount +
              targetClass;

            if (
              targets.data[
                targetIndex
              ] >
              targets.data[
                currentTarget
              ]
            ) {
              targetClass =
                classIndex;
            }
          }

          if (
            predictionClass ===
            targetClass
          ) {
            totalCorrect++;
          }
        }

        totalSamples +=
          samples.length;

        // --------------------------
        // BACKWARD
        // --------------------------

        const gradient =
          softmaxCrossEntropyGradient(
            predictions,
            targets
          );

        this.model.backward(
          gradient,
          this.learningRate
        );
      }

      const averageLoss =
        totalLoss /
        totalSamples;

      const accuracy =
        totalCorrect /
        totalSamples;

      const result = {
        epoch,
        loss: averageLoss,
        accuracy
      };

      history.push(result);

      console.log(
        `Epoch ${epoch}/${this.epochs} ` +
        `| loss: ${averageLoss.toFixed(4)} ` +
        `| accuracy: ${(accuracy * 100).toFixed(2)}%`
      );
    }

    return history;
  }

  createImageBatch(samples) {
    if (samples.length === 0) {
      throw new Error(
        'Cannot create an empty image batch.'
      );
    }

    const firstInput =
      samples[0].input;

    const firstTarget =
      samples[0].target;

    if (
      firstInput.shape.length !== 3
    ) {
      throw new Error(
        'Trainer expects image inputs shaped [channels, height, width].'
      );
    }

    const [
      channels,
      height,
      width
    ] = firstInput.shape;

    const classCount =
      firstTarget.size;

    const imageSize =
      channels *
      height *
      width;

    const inputs =
      new Float32Array(
        samples.length *
        imageSize
      );

    const targets =
      new Float32Array(
        samples.length *
        classCount
      );

    for (
      let i = 0;
      i < samples.length;
      i++
    ) {
      const input =
        samples[i].input;

      const target =
        samples[i].target;

      if (
        input.shape.length !== 3 ||
        input.shape[0] !== channels ||
        input.shape[1] !== height ||
        input.shape[2] !== width
      ) {
        throw new Error(
          'All images in a batch must have the same shape.'
        );
      }

      if (
        target.size !== classCount
      ) {
        throw new Error(
          'All targets in a batch must have the same size.'
        );
      }

      inputs.set(
        input.data,
        i * imageSize
      );

      targets.set(
        target.data,
        i * classCount
      );
    }

    return {
      inputs: new Tensor(
        inputs,
        [
          samples.length,
          channels,
          height,
          width
        ]
      ),

      targets: new Tensor(
        targets,
        [
          samples.length,
          classCount
        ]
      )
    };
  }
}