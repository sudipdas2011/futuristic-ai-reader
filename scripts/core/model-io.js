import fs from "node:fs";
import path from "node:path";

function tensorToArray(tensor) {
  return Array.from(tensor.data);
}

function arrayToFloat32(values) {
  return new Float32Array(values);
}

export function serializeModel(model) {
  return {
    version: 1,

    architecture: {
      type: "CNNReaderV3",
      imageHeight: model.imageHeight,
      imageWidth: model.imageWidth,
      classCount: model.classCount
    },

    conv1: {
      kernels: model.conv1.kernels.map(
        tensorToArray
      ),
      biases: Array.from(model.conv1.biases)
    },

    conv2: {
      kernels: model.conv2.kernels.map(
        tensorToArray
      ),
      biases: Array.from(model.conv2.biases)
    },

    dense: {
      weights: tensorToArray(
        model.dense.weights
      ),
      bias: tensorToArray(
        model.dense.bias
      )
    }
  };
}

export function saveModel(
  model,
  filePath
) {
  const serialized =
    serializeModel(model);

  fs.mkdirSync(
    path.dirname(filePath),
    {
      recursive: true
    }
  );

  fs.writeFileSync(
    filePath,
    JSON.stringify(serialized)
  );

  console.log(
    `Model saved to: ${filePath}`
  );
}

export function loadModelData(
  model,
  data
) {
  for (
    let i = 0;
    i < model.conv1.kernels.length;
    i++
  ) {
    model.conv1.kernels[i].data =
      arrayToFloat32(
        data.conv1.kernels[i]
      );
  }

  model.conv1.biases =
    arrayToFloat32(
      data.conv1.biases
    );

  for (
    let i = 0;
    i < model.conv2.kernels.length;
    i++
  ) {
    model.conv2.kernels[i].data =
      arrayToFloat32(
        data.conv2.kernels[i]
      );
  }

  model.conv2.biases =
    arrayToFloat32(
      data.conv2.biases
    );

  model.dense.weights.data =
    arrayToFloat32(
      data.dense.weights
    );

  model.dense.bias.data =
    arrayToFloat32(
      data.dense.bias
    );

  return model;
}