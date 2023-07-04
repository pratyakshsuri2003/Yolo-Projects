'''This is the code written by pratyaksh.suri. Modifying it without permission is strictly prohibited.
Author: pratyaksh.suri
Email: pratyakshsuri20@gmail.com'''

import io
import torch
from flask import Flask, request, jsonify
from PIL import Image
from flask_cors import CORS
import pandas as pd
import os
import argparse
import logging

df = pd.DataFrame()
df["xmin"] = [50]
df["ymin"] = [50]
df["xmax"] = [50]
df["ymax"] = [50]
df["confidence"] = [0.3]
df["class"] = [4]
df["name"] = [' ']

model = torch.hub.load("F:/Workspace/YOLO V5/yolov5/", 'custom',
                       path=r"F:\Workspace\MODEL FILES .PT\pratyaksh_yolov5_5\best.pt",
                       force_reload=True, source='local')

app = Flask(__name__)
CORS(app)
app.logger.setLevel(logging.INFO)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

@app.route("/test", methods=["GET"])
def check_api():
    return jsonify({"status": 200})

@app.route("/detection", methods=["POST"])
def detect():
    app.logger.info("entered method")
    if not request.method == "POST":
        return jsonify({"error": "Invalid request method"})

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    image_file = request.files["image"]
    image_bytes = image_file.read()
    img = Image.open(io.BytesIO(image_bytes))
    try:
        results = model(img, size=640)  # reduce to 320 for faster inference
        detection = results.pandas().xyxy[0]

        # Convert detection results to a list of dictionaries
        detection_list = detection.to_dict(orient="records")

        response = {
            "detection": detection_list
        }

        app.logger.info("detected")
        return jsonify(response)
    except Exception as e:
        app.logger.error(f'Error {e}')
        return jsonify({"error": "An error occurred during detection"})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    parser.add_argument("--port", default=2003, type=int, help="port number")
    args = parser.parse_args()

    # model = torch.hub.load("F:/Workspace/YOLO V5/yolov5/", 'custom',
    #                        path=r"F:\Workspace\MODEL FILES .PT\pratyaksh_yolov5_5\best.pt",
    #                        force_reload=True, source='local')

    app.run(host="0.0.0.0", port=args.port)
