# config.py
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv(override=True)


def get_bool(env_var, default=False):
    return os.getenv(env_var, str(default)).lower() in ('1', 'true', 'yes', 'on')

# MySQL Configuration
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USERNAME = os.getenv("MYSQL_USERNAME", "graymatics")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "graymatics")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "multi_tenant")

# Restart Check Script Configuration
CHECK_RESTART_DURATION = int(os.getenv("CHECK_RESTART_DURATION", 5))
SCRIPT_NAME = os.getenv("SCRIPT_NAME", "run_step2_sql.py")

# Stream Configuration
HTTP_OUT_IP = os.getenv("HTTP_OUT_IP", "127.0.0.1")
HTTP_OUT_PORT = int(os.getenv("HTTP_OUT_PORT", 8090))
STREAM_IP = os.getenv("STREAM_IP", "127.0.0.1")
STREAM_PORT = int(os.getenv("STREAM_PORT", 8090))

# Frame Sizes
ORIGINAL_WIDTH = int(os.getenv("ORIGINAL_WIDTH", 1280))
ORIGINAL_HEIGHT = int(os.getenv("ORIGINAL_HEIGHT", 720))
STREAMOUT_WIDTH = int(os.getenv("STREAMOUT_WIDTH", 1280))
STREAMOUT_HEIGHT = int(os.getenv("STREAMOUT_HEIGHT", 720))
ORIGINAL_FRAME_SIZE = (ORIGINAL_WIDTH, ORIGINAL_HEIGHT)
STREAMOUT_FRAME_SIZE = (STREAMOUT_WIDTH, STREAMOUT_HEIGHT)
FLAG_VIDEO_LOOP = get_bool("FLAG_VIDEO_LOOP", default=True)
LOG_INTERVAL = int(os.getenv("LOG_INTERVAL", 5))
VID_SKIP_FRAME = int(os.getenv("VID_SKIP_FRAME", 3))
MAX_FPS = int(os.getenv("MAX_FPS", 10))

# ZMQ URL for YOLO
YOLO_ZMQ_URL = os.getenv("YOLO_ZMQ_URL", "tcp://127.0.0.1:4000")


