import { Tensor } from '../../scripts/core/tensor.js';
import { CNNReaderV3 } from '../../scripts/model/cnn-reader-v3.js';

const IMAGE_SIZE = 28;
const CLASS_COUNT = 10;

let model = null;

export function initializeReader() {
  if (!model) {
    model = new CNNReaderV3(
      IMAGE_SIZE, IMAGE_SIZE, CLASS_COUNT
    );
  }

  return model;
}

function canvasToTensor(canvas) {
  const sourceWidth = canvas.width;
  const sourceHeight = canvas.height;

  const sourceContext = canvas.getContext('2d', {
    willReadFrequently: true
  });

  if (!sourceContext) {
    throw new Error(
      'Could not access canvas.'
    );
  }

  const imageData = sourceContext.getImageData(0, 0, sourceWidth, sourceHeight);

  const pixels = new Float32Array(IMAGE_SIZE * IMAGE_SIZE);

  for (
    let y = 0;
    y < IMAGE_SIZE;
    y++
  ) {
    for (
      let x = 0;
      x < IMAGE_SIZE;
      x++
    ) {
      const sourceX = Math.floor( x * sourceWidth / IMAGE_SIZE);
      const sourceY = Math.floor( y * sourceHeight / IMAGE_SIZE);

      const sourceIndex = (sourceY * sourceWidth + sourceX) * 4;

      const red = imageData.data[sourceIndex];
      const green = imageData.data[sourceIndex + 1];
      const blue = imageData.data[sourceIndex + 2];

      const grayscale = ( red + green + blue )/3;

      pixels[ y * IMAGE_SIZE + x ] * grayscale/255;
    }
  }

  return new Tensor(
    pixels,
    [1, IMAGE_SIZE, IMAGE_SIZE]
  );
}

export function predictCanvas(canvas) {
  const reader = initializeReader();
  const image = canvasToTensor(canvas);

  const input = new Tensor( new Float32Array( image.data), [1, 1, IMAGE_SIZE, IMAGE_SIZE]);

  const probabilities = reader.predict(input);
  const values = probabilities.data;

  let bestIndex = 0;

  for (
    let i = 1;
    i < CLASS_COUNT;
    i++
  ) {
    if (
      values[i] > values[bestIndex]
    ) {
      bestIndex = i;
    }
  }

  return {
    prediction : bestIndex,
    confidences: Array.from( values)
  };
}