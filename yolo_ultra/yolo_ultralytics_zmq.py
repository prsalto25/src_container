import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time
import zmq
import numpy as np
import cv2

from ultralytics import YOLO


def _parse_shape(shape_bytes):
    """Parse shape bytes like b'720_1280_3' to integers."""
    try:
        h, w, c = map(int, shape_bytes.decode().split("_"))
        return h, w, c
    except Exception as e:
        print(f"[ERROR] Shape parsing failed: {e}")
        return 0, 0, 0


def _parse_frame(frame_bytes, h, w, c):
    """Convert raw bytes to numpy image."""
    try:
        return np.frombuffer(frame_bytes, dtype=np.uint8).reshape((h, w, c))
    except Exception as e:
        print(f"[ERROR] Frame reshaping failed: {e}")
        return None


def _process_result(result):
    """Convert YOLO result to np.array of [x1, y1, x2, y2, conf, cls]"""
    if not result or len(result) == 0:
        return np.array([[]], dtype=np.float32)

    pred = result[0].boxes
    if pred is None or pred.xyxy is None:
        return np.array([[]], dtype=np.float32)

    boxes = pred.xyxy.cpu().numpy()
    conf = pred.conf.cpu().numpy().reshape(-1, 1)
    cls = pred.cls.cpu().numpy().reshape(-1, 1)
    return np.concatenate([boxes, conf, cls], axis=1).astype(np.float32)


def main():
    # Load YOLO TensorRT model
    # model = YOLO("yolov8n.engine")
    model = YOLO("yolov8n")

    # Setup ZMQ server
    port = 4001  # Customizable
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.connect(f"tcp://127.0.0.1:{port}")

    print("[INFO] YOLO TensorRT ZMQ Server Running on port", port)

    while True:
        try:
            # Receive multipart: [frame_bytes, shape_bytes]
            msg = socket.recv_multipart()
            frame_bytes, shape_bytes = msg

            h, w, c = _parse_shape(shape_bytes)
            if h == 0 or w == 0:
                socket.send_pyobj(np.array([[]], dtype=np.float32))
                continue

            frame = _parse_frame(frame_bytes, h, w, c)
            if frame is None:
                socket.send_pyobj(np.array([[]], dtype=np.float32))
                continue

            # Inference
            prev_time = time.time()
            result = model(frame)
            detections = _process_result(result)

            fps = int(1 / (time.time() - prev_time))
            print(f"[INFO] YOLO Inference FPS: {fps}, Detections: {len(detections)}")

            # Send result
            socket.send_pyobj(detections)

        except KeyboardInterrupt:
            print("[INFO] Server shutting down.")
            socket.close()
            context.term()
            break

        except Exception as e:
            print(f"[ERROR] Unhandled exception: {e}")
            socket.send_pyobj(np.array([[]], dtype=np.float32))


if __name__ == "__main__":
    main()
