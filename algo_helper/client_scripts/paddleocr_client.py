import zmq
import base64
import cv2
import numpy as np
import json


class PaddleOCRClient:
    def __init__(self, url="tcp://localhost:5555"):
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(url)

    def ocr(self, frame):
        success, buffer = cv2.imencode('.png', frame)
        if not success:
            raise ValueError("Failed to encode frame")

        # Convert to base64
        b64_img = base64.b64encode(buffer.tobytes()).decode()

        # Prepare message
        message = {
            "action": "predict-by-bytes",
            "payload": b64_img
        }

        # Send request
        self.socket.send_json(message)

        # Wait for response
        response = self.socket.recv_json()
        data = response.get("data", [[]])
        return data