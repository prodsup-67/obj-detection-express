import "dotenv/config";
import cors from "cors";
import debug from "debug";
import express, { ErrorRequestHandler } from "express";
import multer from "multer";

import {
  loadModel,
  readImageFile,
  predict,
  annotateImage,
  writeImageFile,
  readImageEncoded,
  getClassCounts,
} from "./ml.js";
import { PORT, SERVER_URL } from "./utils/env.js";

let global: { model: any } = { model: null };
const logger = debug("myapp");

const app = express();
const upload = multer({ dest: "uploads/" });
app.use(cors({ origin: false }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));
app.use(express.json());
app.use("/static", express.static("public"));

app.get("/", (req, res) => {
  res.json({ hello: "world" });
});

app.get("/load", async (req, res, next) => {
  try {
    await loadModel(global);
    res.json({ model: "Test loading successfully" });
  } catch (err) {
    next(err);
  }
});

app.post("/upload", upload.single("img"), async (req, res, next) => {
  try {
    if (!global.model) await loadModel(global);
    const contentType = req.file?.mimetype ?? "";
    const filePath = req.file?.path ?? "";
    const imageBitmap = await readImageFile(filePath, contentType);
    if (!global.model) {
      throw new Error("No model loaded");
    }
    const predictions = await predict(imageBitmap, global.model);
    const counts = getClassCounts(predictions);
    annotateImage(imageBitmap, predictions);
    const imageURL = await writeImageFile(imageBitmap);
    res.json({ predictions, imageURL, counts });
  } catch (err) {
    next(err);
  }
});

app.post("/upload_base64", async (req, res, next) => {
  try {
    if (!global.model) await loadModel(global);
    const imageEncoded = req.body.imageEncoded ?? "";
    const imageBitmap = await readImageEncoded(imageEncoded);
    const predictions = await predict(imageBitmap, global.model);
    const counts = getClassCounts(predictions);
    annotateImage(imageBitmap, predictions);
    const imageURL = await writeImageFile(imageBitmap);
    res.json({ predictions, imageURL, counts });
  } catch (err) {
    next(err);
  }
});

// JSON Error Middleware
const jsonErrorHandler: ErrorRequestHandler = (err, req, res, next) => {
  let serializedError = JSON.stringify(err, Object.getOwnPropertyNames(err));
  serializedError = serializedError.replace(/\/+/g, "/");
  serializedError = serializedError.replace(/\\+/g, "/");
  res.status(500).send({
    error: serializedError,
    predictions: [],
  });
};
app.use(jsonErrorHandler);

app.on("mount", () => {
  logger("Here");
});

// Running app
app.listen(PORT, async () => {
  logger(`Listening on port ${PORT}: http://localhost:${PORT}`);
  await loadModel(global);
});
