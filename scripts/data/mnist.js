import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';


const BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/";

const FILES = {
  trainImages: "train-images-idx3-ubyte.gz",
  trainLabels: "train-labels-idx1-ubyte.gz",
  testImages: "t10k-images-idx3-ubyte.gz",
  testLabels: "t10k-labels-idx1-ubyte.gz",
};

const CACHE_DIR = path.resolve(
  process.cwd(),
  "scripts",
  "data",
  "mnist-cache"
);

function ensureCacheDir() {
  fs.mkdirSync(CACHE_DIR, { recursive: true,});
}

async function downloadFile(filename) {
  ensureCacheDir();

  const gzPath = path.join(
    CACHE_DIR, filename
  );

  if (fs.existsSync(gzPath)) {
    return gzPath;
  }

  console.log(`Downloading ${filename}...`);

  const response = await fetch(
    BASE_URL + filename
  );

  if (!response.ok) {
    throw new Error(
      `Failed to download ${filename}: ${response.status}`
    );
  }

  const buffer = Buffer.from(
    await response.arrayBuffer()
  );

  fs.writeFileSync(gzPath, buffer);

  console.log(`Saved ${filename}`);

  return gzPath;
}

function readGzip(filename) {
  const compressed = fs.readFileSync(filename);

  return zlib.gunzipSync(compressed);
}

function parseImages(buffer) {
  const magic = buffer.readUInt32BE(0);
  const count = buffer.readUInt32BE(4);
  const rows = buffer.readUInt32BE(8);
  const cols = buffer.readUInt32BE(12);

  if (magic !== 2051) {
    throw new Error(
      `Invalid MNIST image file. Magic: ${magic}`
    );
  }

  if (rows !== 28 || cols !== 28) {
    throw new Error(
      `Expected 28x28 images, got ${rows}x${cols}`
    );
  }

  const imageSize = rows * cols;

  const expectedSize =
    16 + count * imageSize;

  if (buffer.length < expectedSize) {
    throw new Error(
      "MNIST image file is incomplete."
    );
  }

  return {
    count,
    rows,
    cols,
    data: buffer.subarray(16),
  };
}

function parseLabels(buffer) {
  const magic = buffer.readUInt32BE(0);
  const count = buffer.readUInt32BE(4);

  if (magic !== 2049) {
    throw new Error(
      `Invalid MNIST label file. Magic: ${magic}`
    );
  }

  if (buffer.length < 8 + count) {
    throw new Error(
      "MNIST label file is incomplete."
    );
  }

  return {
    count,
    data: buffer.subarray(8),
  };
}

export async function loadMNIST() {
  ensureCacheDir();

  const [
    trainImagesFile,
    trainLabelsFile,
    testImagesFile,
    testLabelsFile,
  ] = await Promise.all([
    downloadFile(FILES.trainImages),
    downloadFile(FILES.trainLabels),
    downloadFile(FILES.testImages),
    downloadFile(FILES.testLabels),
  ]);

  console.log("Extracting MNIST...");

  const trainImages =
    parseImages(
      readGzip(trainImagesFile)
    );

  const trainLabels =
    parseLabels(
      readGzip(trainLabelsFile)
    );

  const testImages =
    parseImages(
      readGzip(testImagesFile)
    );

  const testLabels =
    parseLabels(
      readGzip(testLabelsFile)
    );

  if (
    trainImages.count !== trainLabels.count
  ) {
    throw new Error(
      "Training image/label count mismatch."
    );
  }

  if (
    testImages.count !== testLabels.count
  ) {
    throw new Error(
      "Test image/label count mismatch."
    );
  }

  console.log(
    `Training samples: ${trainImages.count}`
  );

  console.log(
    `Test samples: ${testImages.count}`
  );

  return {
    train: {
      images: trainImages.data,
      labels: trainLabels.data,
      count: trainImages.count,
      rows: trainImages.rows,
      cols: trainImages.cols,
    },

    test: {
      images: testImages.data,
      labels: testLabels.data,
      count: testImages.count,
      rows: testImages.rows,
      cols: testImages.cols,
    },
  };
}

export function getImage(
  dataset,
  index
) {
  const imageSize =
    dataset.rows * dataset.cols;

  const start =
    index * imageSize;

  return dataset.images.subarray(
    start,
    start + imageSize
  );
}

export function getLabel(
  dataset,
  index
) {
  return dataset.labels[index];
}