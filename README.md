# Tattoo Detection API

A tattoo detection system built using YOLOv8 and FastAPI.
This project exposes a REST API that accepts images and returns tattoo detection results.

The model is a custom-trained YOLOv8s by me.

Link to the training repo: https://github.com/aaghaazkhan/AI_Training/tree/main/Day_3_and_Day_4

---

## Guide to run this repo on your system 

### Step 1: Clone the repo
```bash
git clone https://github.com/aaghaazkhan/tattoo-detection-api.git

cd tattoo-detection-api
```

### Step 2: Install PyTorch
Visit https://pytorch.org/get-started/locally/ and install PyTorch according to:
- OS
- Package
- Compute Platform

### Step 3: Install project dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download the Model Weights

Download the trained YOLOv8 model (.pt) from the link below:

Trained Model Link: https://drive.google.com/drive/u/0/folders/1uER4UQMt0PW4Zn6zMYG3qree6PfQVCPR

After downloading, place the model file in the following dir:
```
src/model/
```

### Step 5: Run the app
Open the terminal and enter the following command:

```bash
uvicorn src.main:app --reload
```

## Traied Model Information

### Dataset Information
- Total Images: 395
    - Train: 385
    - Validation: 10
- Classes: 1 (tattoo)
- Annotation Format: YOLO format
```
class_id x_center y_center width height
```
---
### Data Augmentation Used
    1. HorizontalFlip
    2. VerticalFlip
    3. RandomBrightnessContrast
    4. Rotate
    5. Blur

* Albumentations was used for performing the data augmentation
* Image size used during training: 512 × 512
---
### Model Used

- Model: YOLOv8s (yolov8s.pt)
- Pretrained on: COCO dataset
- Fine-tuned for: Tattoo detection (1 class)
---
### Training Configuration
```
- Epochs: 100 (Early stopped at epoch 26)
- Batch Size: 32
- Optimizer: SGD
- Learning Rate: 0.001
- Image Size: 512
- Device: GPU (CUDA)
```
---
### Hardware Used
- Training was performed locally on:
        NVIDIA RTX 3050 GPU
- VRAM: 6 GB
---
### Checkpoints Generated
---
- During training, YOLO automatically generated:
    1. last.pt - latest epoch weights
    2. best.pt - best validation performance weights
---
### Final evaluation was done using:
- best.pt
---
### Validation Metrics (Best Model)
- Precision: 1.00
- Recall: 0.99
- mAP@50: 0.995
- mAP@50-95: 0.897
---
### Important Links
- Base Model: https://drive.google.com/drive/u/0/folders/1501rdRCBTjN5rMAZ2hl-WiNHsV05JLH_
- Dataset: https://drive.google.com/drive/u/0/folders/1oOz60kMFMVD7Dt2V6hAhLI6Pxv5HyVAD
- Training Checkpoints: https://drive.google.com/drive/u/0/folders/14NiMA-1CrR8_fWbpSmNEQEluMOncLd8M
- Trained Model: https://drive.google.com/drive/u/0/folders/1uER4UQMt0PW4Zn6zMYG3qree6PfQVCPR
---

#### ( NOTE: The validation metrics might be misleading due to less amount of data in the validation split.)