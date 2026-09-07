import { Tensor } from "../../scripts/core/tensor.js";
import { CNNReaderV3 } from "../../scripts/model/cnn-reader-v3.js";
import { loadModelData } from "../../scripts/core/model-io.js";

const IMAGE_SIZE = 28;
const CLASS_COUNT = 10;

let model = null;
let loadingPromise = null;

export async function initializeReader() {
  if (model) return model;

  if (!loadingPromise) {
    loadingPromise = (async () => {
      const response = await fetch(
        `${import.meta.env.BASE_URL}models/cnn-reader-v3.json`
      );

      if (!response.ok) {
        throw new Error(
          `Failed to load trained model: ${response.status}`
        );
      }

      const modelData = await response.json();

      const reader = new CNNReaderV3(
        IMAGE_SIZE,
        IMAGE_SIZE,
        CLASS_COUNT
      );

      loadModelData(reader, modelData);

      model = reader;

      console.log("AI Reader: trained model loaded.");

      return model;
    })();
  }

  return loadingPromise;
}

function canvasToTensor(canvas) {
  const ctx = canvas.getContext("2d", {
    willReadFrequently: true,
  });

  const { width, height } = canvas;

  const image = ctx.getImageData(
    0,
    0,
    width,
    height
  );

  const pixels = new Float32Array(
    IMAGE_SIZE * IMAGE_SIZE
  );

  for (let y = 0; y < IMAGE_SIZE; y++) {
    for (let x = 0; x < IMAGE_SIZE; x++) {
      const sourceX = Math.floor(
        (x / IMAGE_SIZE) * width
      );

      const sourceY = Math.floor(
        (y / IMAGE_SIZE) * height
      );

      const index =
        (sourceY * width + sourceX) * 4;

      const r = image.data[index];
      const g = image.data[index + 1];
      const b = image.data[index + 2];

      const grayscale =
        (r + g + b) / 3 / 255;

      // Canvas: dark digit on light background
      // MNIST: bright digit on dark background
      const value = 1 - grayscale;

      pixels[y * IMAGE_SIZE + x] = value;
    }
  }

  return new Tensor(
    pixels,
    [1, 1, IMAGE_SIZE, IMAGE_SIZE]
  );
}

export async function predictCanvas(canvas) {
  const reader = await initializeReader();

  const input = canvasToTensor(canvas);

  const probabilities =
    reader.predict(input);

  const values = Array.from(
    probabilities.data
  );

  let bestIndex = 0;

  for (let i = 1; i < values.length; i++) {
    if (values[i] > values[bestIndex]) {
      bestIndex = i;
    }
  }

  return {
    digit: bestIndex,
    confidence: values[bestIndex],
    probabilities: values,
  };
}