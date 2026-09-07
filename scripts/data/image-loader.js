import sharp from 'sharp';
import { Tensor } from '../core/tensor.js';

export async function loadImage(
  filePath,
  options = {}
) {
  const {
    width = 28,
    height = 28
  } = options;

  const { data, info } =
    await sharp(filePath)
      .resize(width, height, {
        fit: 'fill'
      })
      .grayscale()
      .raw()
      .toBuffer({
        resolveWithObject: true
      });

  if (
    info.width !== width ||
    info.height !== height
  ) {
    throw new Error(
      `Expected ${width}×${height} image, ` +
      `received ${info.width}×${info.height}.`
    );
  }

  const pixels =
    new Float32Array(
      width * height
    );

  for (
    let i = 0;
    i < pixels.length;
    i++
  ) {
    pixels[i] =
      data[i] / 255;
  }

  return new Tensor(
    pixels,
    [1, height, width]
  );
}