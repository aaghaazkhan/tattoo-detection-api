# =========================================
# Imports
# =========================================
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from src.inference import run_inference # our custom function for tattoo detection
OUTPUT_DIR = r"C:\Users\Aaghaaz Khan\Personal\NoteActive\AI_Training\runs\detect\src\outputs\predict"
CROP_OUTPUT_DIR = r"C:\Users\Aaghaaz Khan\Personal\NoteActive\AI_Training\runs\detect\src\outputs\predict\crop"


# =========================================
# App Initialization
# =========================================
app = FastAPI(title="Tattoo Detection API")

# =========================================
# Routes
# =========================================
@app.get('/')

def index():
    return "Visit /docs for using this app"

#--------------------------------------------------------------

@app.post('/predict')

def predict(image: UploadFile = File(..., description="Upload a tattoo image")):

    input_dir = "src/inputs"
    os.makedirs(input_dir, exist_ok=True)
    image_path = os.path.join(input_dir, image.filename)

    with open(image_path, 'wb') as f:
        f.write(image.file.read())

    image.file.close()
    
    output_image_path, status, confidence_score, crop_output_image_path = run_inference(image_path)
    filename = os.path.basename(output_image_path)
    crop_filename = os.path.basename(crop_output_image_path)

    return {
        "tattoo_status": status,
        "confidence_score": confidence_score,
        "output_url": f"http://127.0.0.1:8000/output_file/{filename}",
        "crop_output_url": f"http://127.0.0.1:8000/crop_output_file/{crop_filename}"
    }

#--------------------------------------------------------------

@app.get("/output_file/{filename}")

def output_file(filename: str):

    file_path = os.path.join(OUTPUT_DIR, filename)
    return FileResponse(
        file_path,
        media_type="image/jpeg"
    )

@app.get("/crop_output_file/{filename}")

def crop_output_file(filename: str):
    crop_file_path = os.path.join(CROP_OUTPUT_DIR, filename)
    return FileResponse(
        crop_file_path,
        media_type="image/jpeg"
    )