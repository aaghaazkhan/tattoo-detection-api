# =========================================
# Imports
# =========================================
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from src.inference import run_inference # our custom function for tattoo detection

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
    
    output_image_path = run_inference(image_path)

    return FileResponse(
        output_image_path,
        media_type='image/jpeg',
        filename='prediction.jpeg'
    )