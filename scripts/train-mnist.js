import path from "node:path";
import { fileURLToPath } from "node:url";

import { loadMNIST } from "./data/mnist.js";
import { CNNReaderV3 } from "./model/cnn-reader-v3.js";
import { Trainer } from "./core/trainer.js";
import { saveModel } from "./core/model-io.js";

const __filename =
  fileURLToPath(import.meta.url);

const __dirname =
  path.dirname(__filename);

async function main() {
  console.log("");
  console.log("================================");
  console.log("       FUTURISTIC AI READER");
  console.log("          MNIST TRAINER");
  console.log("================================");
  console.log("");

  // -----------------------------
  // LOAD MNIST
  // -----------------------------

  console.log("Loading MNIST...");

  const dataset =
    await loadMNIST();

  console.log("");
  console.log(
    `Training samples: ${dataset.train.count}`
  );

  console.log(
    `Testing samples:  ${dataset.test.count}`
  );

  // -----------------------------
  // CREATE CNN
  // -----------------------------

  console.log("");
  console.log("Creating CNNReaderV3...");

  const model =
    new CNNReaderV3(
      28,
      28,
      10
    );

  console.log("Model ready.");

  // -----------------------------
  // TRAIN
  // -----------------------------

  const trainer =
    new Trainer(
      model,
      {
        learningRate: 0.001,
        epochs: 1,
        batchSize: 32,
        shuffle: true
      }
    );

  const history =
    trainer.train(
      dataset.train
    );

  // -----------------------------
  // SAVE MODEL
  // -----------------------------

  const modelPath =
    path.join(
      __dirname,
      "..",
      "public",
      "models",
      "cnn-reader-v3.json"
    );

  console.log("");
  console.log("Saving trained model...");

  saveModel(
    model,
    modelPath
  );

  // -----------------------------
  // RESULTS
  // -----------------------------

  console.log("");
  console.log("================================");
  console.log("         TRAINING COMPLETE");
  console.log("================================");

  for (
    const result of history
  ) {
    console.log(
      `Epoch ${result.epoch} | ` +
      `loss ${result.loss.toFixed(4)} | ` +
      `accuracy ${(
        result.accuracy * 100
      ).toFixed(2)}% | ` +
      `${result.seconds}s`
    );
  }

  console.log("");
  console.log(
    `Model: ${modelPath}`
  );

  console.log("");
}

main().catch((error) => {
  console.error("");
  console.error(
    "================================"
  );
  console.error(
    "          TRAINING FAILED"
  );
  console.error(
    "================================"
  );
  console.error("");
  console.error(error);
  process.exit(1);
});