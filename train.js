import * as tf from '@tensorflow/tfjs-node';
import mnist from 'mnist';
import fs from 'fs';
import path from 'path';

async function executeLocalTraining() {
  console.log('--- INITIALIZING AI CORE ENGINE TRAINING PHASE ---');

  // 1. Download and format 8,000 raw training pairs and 2,000 validation pairs
  const set = mnist.set(8000, 2000);
  
  const trainImages = [];
  const trainLabels = [];
  set.training.forEach(item => {
    trainImages.push(item.input);
    trainLabels.push(item.output);
  });

  const testImages = [];
  const testLabels = [];
  set.test.forEach(item => {
    testImages.push(item.input);
    testLabels.push(item.output);
  });

  // Convert array datasets into optimized tensor chunks matching 28x28 grayscale inputs
  const xs = tf.tensor4d(trainImages, [trainImages.length, 28, 28, 1]);
  const ys = tf.tensor2d(trainLabels, [trainLabels.length, 10]);
  const testXs = tf.tensor4d(testImages, [testImages.length, 28, 28, 1]);
  const testYs = tf.tensor2d(testLabels, [testLabels.length, 10]);

  // 2. Structural Layer Architecture Definition Configuration
  const model = tf.sequential();
  
  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    kernelSize: 3,
    filters: 16,
    activation: 'relu'
  }));
  
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 10, activation: 'softmax' })); // Outputs 0-9 probabilities

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  // 3. Execution Node Training Process
  console.log('COMMENCING OPTIMIZATION MATRICES EPON COUNTER...');
  await model.fit(xs, ys, {
    epochs: 10,
    batchSize: 128,
    validationData: [testXs, testYs],
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch + 1}: Loss = ${logs.loss.toFixed(4)} | Acc = ${(logs.acc * 100).toFixed(2)}%`);
      }
    }
  });

  // 4. Save the compiled structural weights directly into Vite's public assets folder
  const outputDirectory = path.join(process.cwd(), 'public', 'model');
  if (!fs.existsSync(outputDirectory)){
      fs.mkdirSync(outputDirectory, { recursive: true });
  }

  await model.save(`file://${outputDirectory}`);
  console.log(`\nSUCCESS: Perfect weight mapping sheets exported to: ${outputDirectory}`);
  process.exit(0);
}

executeLocalTraining();
