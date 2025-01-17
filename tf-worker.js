const workerpool = require("workerpool");
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");

// Function to classify a single image
async function classifyImage(filePath) {
  const imageBuffer = fs.readFileSync(filePath);
  const imageTensor = tf.node.decodeImage(imageBuffer);

  try {
    const predictions = await model.classify(imageTensor);
    return predictions;
  } finally {
    imageTensor.dispose();
  }
}

// Expose the classifyImage function to the workerpool
workerpool.worker({
  classifyImage,
});
