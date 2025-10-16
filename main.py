# Importing FastAPI dependencies
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

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

# Initialize FastAPI
app = FastAPI()

#Mount Templates Directory
templates = Jinja2Templates(directory="templates")

# Load face detection model
model = FaceAnalysis(name="buffalo_l", root='C:/Users/muhammadannasasif/.insightface')
model.prepare(ctx_id=0)

# Load Swapper Model
swapper = insightface.model_zoo.get_model('models/inswapper_128.onnx', download=False, download_zip=False)

