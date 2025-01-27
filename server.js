const express = require("express");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const nsfw = require("nsfwjs");
const path = require("path");
const fs = require("fs");
const sharp = require("sharp");
const ffmpeg = require("fluent-ffmpeg");
const ffmpegPath = "./ffmpeg/ffmpeg-7.0.2-amd64-static/ffmpeg";
ffmpeg.setFfmpegPath(ffmpegPath);
const workerpool = require("workerpool");
const rateLimit = require("express-rate-limit");
let pLimit;

(async () => {
  pLimit = (await import("p-limit")).default;

  console.log("Loading NSFW model...");
  const model = await nsfw.load();
  console.log("NSFW model loaded");

  const app = express();
  const PORT = 5000;
  const pool = workerpool.pool(__dirname + "/tf-worker.js");

  const upload = multer({ dest: "uploads/" });

  const limiter = rateLimit({
    windowMs: 1 * 60 * 1000, // 1 minute
    max: 20, // Limit each IP to 20 requests per window
  });
  app.use(limiter);

  async function extractFrames(videoPath, outputDir, frameRate = 1) {
    return new Promise((resolve, reject) => {
      fs.mkdirSync(outputDir, { recursive: true });
      ffmpeg(videoPath)
        .outputOptions([`-vf fps=${frameRate}`])
        .on("end", () => {
          console.log("Frame extraction complete.");
          resolve(
            fs.readdirSync(outputDir).map((file) => path.join(outputDir, file))
          );
        })
        .on("stderr", (stderrLine) => {
          console.log(`FFmpeg stderr: ${stderrLine}`);
        })
        .on("error", (err) => {
          console.log(`FFmpeg error: ${err.message}`);
          reject(err);
        })
        .output(`${outputDir}/frame-%04d.jpg`)
        .run();
    });
  }

  async function resizeImage(filePath, maxWidth = 800, maxHeight = 800) {
    const dir = path.dirname(filePath);
    const ext = path.extname(filePath);
    const baseName = path.basename(filePath, ext);
    const resizedPath = path.join(dir, `${baseName}_resized${ext}`);

    await sharp(filePath)
      .resize({ width: maxWidth, height: maxHeight, fit: "inside" })
      .toFile(resizedPath);

    return resizedPath;
  }

  async function resizeFrame(filePath, maxWidth = 800, maxHeight = 800) {
    return await resizeImage(filePath, maxWidth, maxHeight);
  }

  async function processImages(files) {
    try {
      const results = await Promise.all(
        files.map(async (file) => {
          const resizedPath = await resizeImage(file.path);
          const imageBuffer = fs.readFileSync(resizedPath);
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
            imageTensor.dispose();
            fs.unlinkSync(file.path);
            fs.unlinkSync(resizedPath);
          }
        })
      );

      return results;
    } catch (error) {
      files.forEach((file) => fs.unlinkSync(file.path));
      throw error;
    }
  }

  async function processVideo(file) {
    const framesDir = path.join(
      "uploads",
      `${path.basename(file.path, path.extname(file.originalname))}_frames`
    );

    try {
      const framePaths = await extractFrames(file.path, framesDir, 1);
      const limit = pLimit(5);

      const results = await Promise.all(
        framePaths.map((framePath) =>
          limit(async () => {
            const resizedPath = await resizeFrame(framePath);
            const frameBuffer = fs.readFileSync(resizedPath);
            const frameTensor = tf.node.decodeImage(frameBuffer);

            try {
              const predictions = await model.classify(frameTensor);
              const pornPrediction = predictions.find(
                (p) => p.className === "Porn"
              );
              return pornPrediction && pornPrediction.probability > 0.5;
            } finally {
              frameTensor.dispose();
              fs.unlinkSync(resizedPath);
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

  app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
  });
})();
