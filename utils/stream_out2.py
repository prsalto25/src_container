from flask import Flask, Response
import threading
import cv2
import re

class OutStream:
    def __init__(self, out_xy, output_path):
        self.out_xy = out_xy
        self.output_path = output_path
        self.frame = None

        # Extract port from output_path like feed10.ffm → 9010
        match = re.search(r'feed(\d+)\.ffm', output_path)
        if not match:
            raise ValueError("Invalid output_path format. Expected format: feed<number>.ffm")
        self.port = 9000 + int(match.group(1))

        self.app = Flask(__name__)
        self._setup_routes()

        # Start Flask server in a background thread
        threading.Thread(target=self._run_server, daemon=True).start()

    def _setup_routes(self):
        @self.app.route('/')
        def stream():
            return Response(self._generate(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

    def _run_server(self):
        self.app.run(host='0.0.0.0', port=self.port, threaded=True)

    def _generate(self):
        while True:
            if self.frame is None:
                continue
            resized = cv2.resize(self.frame, self.out_xy)
            _, buffer = cv2.imencode('.jpg', resized)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def write(self, frame):
        self.frame = frame
