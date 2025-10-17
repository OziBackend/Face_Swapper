# Importing FastAPI dependencies
from fastapi import FastAPI, Request, HTTPException, File, UploadFile, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
#======================================================
# Importing Python dependencies
import os
import json
import numpy as np
import cv2
import threading
import asyncio
import glob
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import pillow_heif
#======================================================
# Importing InsightFace dependencies
import insightface
from insightface.app import FaceAnalysis
#======================================================
# Importing config
import environment.config as config
from environment.messages import SERVER_RUNNING
#======================================================
# Importing controller dependencies
import controller.controller as controller


#=============================================================================================
#===================================INITIALIZATION============================================
#=============================================================================================
# Initialize FastAPI
app = FastAPI()

#Mount Templates Directory
templates = Jinja2Templates(directory=config.TEMPLATES_DIR)

# Mount Static Directory
app.mount(f"/{config.STATIC_DIR}", StaticFiles(directory=config.STATIC_DIR), name="static")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#create threadpool
executor = ThreadPoolExecutor(max_workers=4)

#load pillow_heif (to handle HEIC images)
pillow_heif.register_heif_opener()

#=============================================================================================
# Model Loading
# Load face detection model
model = FaceAnalysis(name=config.DETECTION_MODEL_NAME, root=config.DETECTION_MODEL_ROOT)
model.prepare(ctx_id=config.DETECTION_MODEL_CTX_ID)

# Load Swapper Model
swapper = insightface.model_zoo.get_model(config.MODEL_PATH, download=False, download_zip=False)


#=============================================================================================
#======================================ROUTES=================================================
#=============================================================================================

# Routes
#Server Checking Route
@app.get("/")
def index(request: Request):
    return JSONResponse({"message": SERVER_RUNNING})

#==========================================================

#Face Swapping Route
@app.post("/swap_face")
async def swap_face(
    request: Request,
    file: UploadFile = File(...)
):
    # Read file from request
    contents = await file.read()
    image = Image.open(BytesIO(contents))
    image = image.convert("RGB")
    
    # Convert PIL image to cv2 image (numpy array)
    image_array = np.array(image)
    image_copy = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor, 
        controller.face_swap_func, 
        image_copy, swapper, model, file.filename
    )
    return result

#==========================================================
# Read Image Route
@app.get("/read_image")
def read_image_route(request: Request):
    return controller.read_image(request)