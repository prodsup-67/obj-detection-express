import debug from "debug";
import tf from "@tensorflow/tfjs";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "dotenv/config";
import fs from "fs";
// @ts-ignore
import * as PImage from "pureimage";
import sharp from "sharp";
import { Readable } from "stream";
import promisify from "util.promisify";
import { PORT, SERVER_URL } from "./utils/env";

const logger = debug("myapp");
const readFile = promisify(fs.readFile);
const unlinkFile = promisify(fs.unlink);
const readDirFile = promisify(fs.readdir);

const font = PImage.registerFont(
  "src/utils/fonts/NotoSans-Medium.ttf",
  "NotoSans"
);
font.loadSync();

export async function loadModel(global: any) {
  await tf.ready();
  const modelUrl = `http://localhost:${PORT}/static/coco-ssd/model.json`;
  const model = await cocoSsd.load({
    base: "lite_mobilenet_v2", //Fastest
    // base: "mobilenet_v2", // Default
    modelUrl: modelUrl,
  });
  global.model = model;
  logger("Load model successfully");
}

export const bufferToStream = (binary: Buffer) => {
  const readableInstanceStream = new Readable({
    read() {
      this.push(binary);
      this.push(null);
    },
  });

  return readableInstanceStream;
};

export async function readImageFile(filePath: string, contentType: string) {
  let buffer = await readFile(filePath);
  let bufferPNG = await sharp(buffer).resize(300).toFormat("png").toBuffer();
  // let bufferPNG = await sharp(buffer).toFormat("png").toBuffer();
  const stream = bufferToStream(bufferPNG);
  const imageBitmap = await PImage.decodePNGFromStream(stream);
  return imageBitmap;
}

export const predict = async (imageBitmap: any, model: any) => {
  const predictions = await model.detect(imageBitmap);
  return predictions as cocoSsd.DetectedObject[];
};

export const getClassCounts = (predictions: cocoSsd.DetectedObject[]) => {
  const countsObj = {} as any;

  predictions.forEach((pred) => {
    const val = countsObj?.[pred.class] ?? 0;
    countsObj[pred.class] = val + 1;
  });

  const countsArr: any[] = [];
  for (const [key, value] of Object.entries(countsObj)) {
    countsArr.push({ class: key, count: value });
  }

  countsArr.sort(function (a, b) {
    const na = a.count;
    const nb = b.count;
    if (na < nb) return 1;
    if (nb > na) return -1;
    return 0;
  });

  return { countsObj, countsArr };
};

export function annotateImage(
  imageBitmap: any,
  predictions: cocoSsd.DetectedObject[]
) {
  // Validation
  const ctx = imageBitmap.getContext("2d");
  if (!ctx) return;

  ctx.clearRect(0, 0, ctx.width, ctx.height);
  if (predictions.length > 0) {
    predictions.forEach((prediction) => {
      if (prediction.score > 0) {
        drawBox(prediction, ctx);
      }
    });
  }
}

function drawBox(prediction: cocoSsd.DetectedObject, ctx: any) {
  let bboxLeft = prediction.bbox[0];
  let bboxTop = prediction.bbox[1];
  let bboxWidth = prediction.bbox[2];
  let bboxHeight = prediction.bbox[3]; // - bboxTop;

  ctx.beginPath();
  ctx.font = "12px NotoSans";
  ctx.fillStyle = "red";

  ctx.fillText(
    prediction.class + ": " + Math.round(prediction.score * 100) + "%",
    bboxLeft + 5,
    bboxTop + 30
  );

  ctx.rect(bboxLeft, bboxTop, bboxWidth, bboxHeight);
  ctx.strokeStyle = "#FF0000";
  ctx.fillStyle = "rgba(140, 41, 162, 0.2)";
  ctx.lineWidth = 3;
  ctx.stroke();
  ctx.fill();
}

export async function writeImageFile(imageBitmap: any) {
  // Remove all png files in the public folder
  const files = await readDirFile("public");
  const filesPng = files.filter((f) => f.includes(".png"));
  for (const file of filesPng) {
    await unlinkFile(`public/${file}`);
  }
  // Write a new file
  const timestamp = new Date().getTime();
  const filename = `output_${timestamp}.png`;
  const imageURL = `${SERVER_URL}/static/${filename}`;
  await PImage.encodePNGToStream(
    imageBitmap,
    fs.createWriteStream(`public/${filename}`)
  );
  return imageURL;
}

export async function readImageEncoded(imageEncoded: string) {
  let buffer = Buffer.from(imageEncoded, "base64");
  let bufferPNG = await sharp(buffer).toFormat("png").toBuffer();
  const stream = bufferToStream(bufferPNG);
  const imageBitmap = await PImage.decodePNGFromStream(stream);
  return imageBitmap;
}
