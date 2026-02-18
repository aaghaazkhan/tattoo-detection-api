# =========================================
# Imports
# =========================================
from ultralytics import YOLO
from pathlib import Path
import cv2
import torch
import os

# ========================================
# Load the trained YOLOv8s model
# ========================================
model_path = "src/model/tattoo_yolov8s.pt"
model = YOLO(model_path)


# ========================================
# Runs prediction on unseen images
# ========================================
def run_inference(image_path):

    result = model.predict(
    source=image_path,
    conf=0.6,
    iou=0.5,
    save=True,
    # project=ROOT_DIR/"src"/"outputs", # base output folder
    project="src/outputs",
    name="", # subfolder (NONE here)
    exist_ok=True
    )

    save_dir = Path(result[0].save_dir)
    output_image_path = save_dir / Path(image_path).name

    r = result[0]

    status = "True"
    if r.boxes is None:
        status = "False"

    confidence_score = float(r.boxes.conf[0])

    return output_image_path, status, confidence_score