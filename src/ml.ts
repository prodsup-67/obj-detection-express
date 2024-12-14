import debug from "debug";
// import tf from "@tensorflow/tfjs";
const tf = require("@tensorflow/tfjs");
require("@tensorflow/tfjs-node");
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "dotenv/config";
import fs from "fs";
// @ts-ignore
import * as PImage from "pureimage";
import sharp from "sharp";
import { Readable } from "stream";
import promisify from "util.promisify";
import { PORT } from "./utils/env";

const logger = debug("myapp");
const readFile = promisify(fs.readFile);

const font = PImage.registerFont(
  "src/utils/fonts/NotoSans-Medium.ttf",
  "NotoSans"
);
font.loadSync();

export async function loadModel(global: any) {
  await tf.ready();
  const modelUrl = `http://localhost:${PORT}/static/coco-ssd/model.json`;
  const model = await cocoSsd.load({ modelUrl: modelUrl });
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

export async function writeImageFile(imageBitmap: any, filePath: string) {
  await PImage.encodePNGToStream(imageBitmap, fs.createWriteStream(filePath));
}
export async function readImageEncoded(imageEncoded: string) {
  let buffer = Buffer.from(imageEncoded, "base64");
  let bufferPNG = await sharp(buffer).toFormat("png").toBuffer();
  const stream = bufferToStream(bufferPNG);
  const imageBitmap = await PImage.decodePNGFromStream(stream);
  return imageBitmap;
}
