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

    r = result[0]
    confidence_score = 0.0
    status = False
    crop_output_image_path = None
    output_image_path = None

    if r.boxes is not None and len(r.boxes) > 0:
        status = True

        box = r.boxes.xyxy[0]
        coords = []

        for value in box:
            coords.append(int(value))

        x1, y1, x2, y2 = coords

        img = cv2.imread(image_path)
        crop_img = img[y1:y2, x1:x2] # img[y1:y2, x1:x2]

        save_dir = Path(result[0].save_dir)
        crop_save_dir = save_dir/"crop"
        crop_save_dir.mkdir(exist_ok=True)

        output_image_path = save_dir / Path(image_path).name
        crop_output_image_path = crop_save_dir / Path(image_path).name
        cv2.imwrite(crop_output_image_path, crop_img)

        confidence_score = float(r.boxes.conf[0])

    return output_image_path, status, confidence_score, crop_output_image_path