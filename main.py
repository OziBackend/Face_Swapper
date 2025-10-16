# Importing FastAPI dependencies
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Importing Python dependencies
import os
import json
import numpy as np
import cv2
import threading
import glob
from PIL import Image

# Importing InsightFace dependencies
import insightface
from insightface.app import FaceAnalysis

# Importing config
from environment.config import *

# Initialize FastAPI
app = FastAPI()

#Mount Templates Directory
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Mount Static Directory
app.mount(STATIC_DIR, StaticFiles(directory="static"), name="static")

# Load face detection model
model = FaceAnalysis(name=DETECTION_MODEL_NAME, root=DETECTION_MODEL_ROOT)
model.prepare(ctx_id=DETECTION_MODEL_CTX_ID)

# Load Swapper Model
swapper = insightface.model_zoo.get_model(MODEL_PATH, download=False, download_zip=False)

#==========================================================
# Routes

@app.get("/")
def index(request: Request):
    return JSONResponse({"message": "Face Swapper Service is running"})