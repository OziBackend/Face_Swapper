PORT = 9002
IP = "172.16.0.94"

MODEL_PATH = "models/inswapper_128.onnx"
DETECTION_MODEL_NAME = "buffalo_l"
DETECTION_MODEL_ROOT = "C:/Users/muhammadannasasif/.insightface"
DETECTION_MODEL_CTX_ID = 0

TEMPLATES_DIR = "templates"
STATIC_DIR = "static"
STATIC_PATH = "static"

IMAGE_URL_PREFIX = f"http://{IP}:{PORT}/read_image?file_name="