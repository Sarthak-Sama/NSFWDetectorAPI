const express = require("express");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const nsfw = require("nsfwjs");
const path = require("path");
const fs = require("fs");
const ffmpeg = require("fluent-ffmpeg");
const workerpool = require("workerpool");
const rateLimit = require("express-rate-limit");
let pLimit;

(async () => {
  pLimit = (await import("p-limit")).default;

  // Initialize the model at server startup
  console.log("Loading NSFW model...");
  const model = await nsfw.load();
  console.log("NSFW model loaded");

  const app = express();
  const PORT = 5000;
  const pool = workerpool.pool(__dirname + "/tf-worker.js");

  // Configure Multer for file uploads
  const upload = multer({ dest: "uploads/" });

  // Rate limiter
  const limiter = rateLimit({
    windowMs: 1 * 60 * 1000, // 1 minute
    max: 20, // Limit each IP to 20 requests per window
  });
  app.use(limiter);

  // Utility to extract frames from a video
  async function extractFrames(videoPath, outputDir, frameRate = 1) {
    return new Promise((resolve, reject) => {
      fs.mkdirSync(outputDir, { recursive: true });
      ffmpeg(videoPath)
        .outputOptions([`-vf fps=${frameRate}`])
        .on("end", () =>
          resolve(
            fs.readdirSync(outputDir).map((file) => path.join(outputDir, file))
          )
        )
        .on("error", (err) => reject(err))
        .output(`${outputDir}/frame-%04d.jpg`)
        .run();
    });
  }

  // Function to process multiple images
  async function processImages(files) {
    try {
      const results = await Promise.all(
        files.map(async (file) => {
          const imageBuffer = fs.readFileSync(file.path);
          const imageTensor = tf.node.decodeImage(imageBuffer);

          try {
            const predictions = await model.classify(imageTensor);
            const pornPrediction = predictions.find(
              (p) => p.className === "Porn"
            );
            const hentaiPrediction = predictions.find(
              (p) => p.className === "Hentai"
            );

            const isNSFW =
              (pornPrediction && pornPrediction.probability > 0.5) ||
              (hentaiPrediction && hentaiPrediction.probability > 0.5);

            return {
              filename: file.originalname,
              result: isNSFW ? "NSFW" : "SFW",
              probability: pornPrediction ? pornPrediction.probability : 0,
            };
          } finally {
            imageTensor.dispose(); // Dispose of the tensor to free memory
          }
        })
      );

      return results;
    } finally {
      files.forEach((file) => fs.unlinkSync(file.path)); // Clean up the uploaded files
    }
  }

  // Process video frames
  async function processVideo(file) {
    const framesDir = path.join(
      "uploads",
      `${path.basename(file.path, path.extname(file.originalname))}_frames`
    );

    try {
      const framePaths = await extractFrames(file.path, framesDir, 1); // 1 frame/sec
      const limit = pLimit(5); // Limit concurrent operations to 5

      const results = await Promise.all(
        framePaths.map((framePath) =>
          limit(async () => {
            const frameBuffer = fs.readFileSync(framePath);
            const frameTensor = tf.node.decodeImage(frameBuffer);

            try {
              const predictions = await model.classify(frameTensor);
              const pornPrediction = predictions.find(
                (p) => p.className === "Porn"
              );
              return pornPrediction && pornPrediction.probability > 0.5;
            } finally {
              frameTensor.dispose();
              fs.unlinkSync(framePath);
            }
          })
        )
      );

      const isNSFW = results.some((result) => result);
      return {
        filename: file.originalname,
        result: isNSFW ? "NSFW" : "SFW",
      };
    } finally {
      fs.rmdirSync(framesDir, { recursive: true });
      fs.unlinkSync(file.path);
    }
  }

  // Endpoint for detecting NSFW content
  app.post("/detect", upload.array("files", 5), async (req, res) => {
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({ error: "No files uploaded" });
    }

    try {
      const results = await Promise.all(
        req.files.map(async (file) => {
          const ext = path.extname(file.originalname).toLowerCase();

          if ([".jpg", ".jpeg", ".png"].includes(ext)) {
            const result = await processImages([file]);
            return result[0];
          } else if ([".mp4", ".mkv", ".avi"].includes(ext)) {
            const result = await processVideo(file);
            return result;
          } else {
            fs.unlinkSync(file.path);
            return {
              filename: file.originalname,
              result: "Unsupported file type",
            };
          }
        })
      );

      res.json({
        totalFiles: results.length,
        results: results,
      });
    } catch (error) {
      req.files.forEach((file) => {
        if (fs.existsSync(file.path)) {
          fs.unlinkSync(file.path);
        }
      });

      res
        .status(500)
        .json({ error: "Error processing files", details: error.message });
    }
  });

  // Start the server only after model is loaded
  app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
  });
})();
