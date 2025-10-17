CURRENT_SYSTEM_IP = "172.16.0.94"
CURRENT_SYSTEM_USER_DIRECTORY = "C:/Users/muhammadannasasif/"

#Port and IP for the server
PORT = 9002
IP = CURRENT_SYSTEM_IP

#Model Paths
MODEL_PATH = "models/inswapper_128.onnx"
DETECTION_MODEL_NAME = "buffalo_l"
DETECTION_MODEL_ROOT = f"{CURRENT_SYSTEM_USER_DIRECTORY}.insightface"
DETECTION_MODEL_CTX_ID = 0

#Templates and Static Files Paths
TEMPLATES_DIR = "templates"
STATIC_DIR = "static"
STATIC_PATH = "static"

#Image URL Prefix
IMAGE_URL_PREFIX = f"http://{IP}:{PORT}/read_image?file_name="