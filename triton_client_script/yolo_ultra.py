import cv2
import sys
import zmq
import numpy as np
from ultralytics import YOLO
from tracking.tracker.byte_tracker_yolox import BYTETracker


class YoloUltralyticsClientZMQ:
    """
    A client to interact with a YOLOv8 model via ZMQ for inference.
    """
    def __init__(self, name="yolov8m", zmq_url="tcp://127.0.0.1:4000", width=640, height=640):
        self.input_height, self.input_width = height, width
        self.model_name = name
        self.zmq_client = self._prepare_zmq_client(zmq_url)
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
                            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
                            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
                            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
                            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
                            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 
                            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 
                            'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
                            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
                            'teddy bear', 'hair drier', 'toothbrush']

        # Initialize trackers for selected classes
        self.class_groups = ['person', 'vehicle', 'bicycle', 'animal', 'object', 'bag']
        self.trackers = {
            cls: BYTETracker() for cls in self.class_groups
        }

    def _prepare_zmq_client(self, zmq_url):
        """
        Prepares the ZMQ client to communicate with the ZMQ server.
        """
        context = zmq.Context()
        socket = context.socket(zmq.REQ)  # Use REQ (Request) socket to send requests and receive responses
        socket.connect(zmq_url)
        return socket

    def _detect_only(self, frame):
        """
        Sends the frame to the ZMQ server for inference and retrieves the detection results.
        """
        frame_bytes = frame.tobytes()
        shape_bytes = f"{frame.shape[0]}_{frame.shape[1]}_{frame.shape[2]}".encode()

        # Send frame and shape to ZMQ server
        self.zmq_client.send_multipart([frame_bytes, shape_bytes])

        # Receive inference result from ZMQ server
        detections = self.zmq_client.recv_pyobj()

        return detections

    def detect(self, frame):
        """
        Main detection.
        """
        detections = self._detect_only(frame)
        return self._format(detections)

    def track(self, detections):
        """
        Tracks detected objects across frames using ByteTrack.
        """
        tracked_results = {}
        for cls, tracker in self.trackers.items():
            tracked_results[cls] = tracker.update(detections.get(cls, []))
        return tracked_results

    def _format(self, detections, mot=False):
        """
        Formats detections into a structured dictionary. (FOR TRACKER PURPOSE)
        """
        formatted_detections = {cls: [] for cls in self.class_groups}
        for x1, y1, x2, y2, conf, cls_id in detections:
            conf *= 100
            class_name = self.class_names[int(cls_id)]

            # Create detection entry
            detection = {
                'conf': conf,
                'cls': class_name,
                'xyxy': [x1, y1, x2, y2],
                'xywh': [int((x1 + x2) / 2), int((y1 + y2) / 2), int(x2 - x1), int(y2 - y1)]
            } if not mot else [int(x1), int(y1), int(x2), int(y2), conf, cls_id]

            # Append to respective class if confidence threshold is met
            if class_name == 'person' and conf > 35:
                formatted_detections['person'].append(detection)
            elif class_name in ['car', 'motorcycle', 'bus', 'truck', 'airplane'] and conf > 20:
                formatted_detections['vehicle'].append(detection)
            elif class_name == 'bicycle' and conf > 20:
                formatted_detections['bicycle'].append(detection)
            elif class_name in ['cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'] and conf > 10:
                formatted_detections['animal'].append(detection)
            # elif class_name in ['backpack', 'suitcase', 'handbag'] and conf > 15:
            #     formatted_detections['object'].append(detection)
            elif class_name in ['backpack', 'handbag', 'backpack', 'suitcase', 'handbag'] and conf > 5:
                formatted_detections['bag'].append(detection)

        return formatted_detections


if __name__ == "__main__":
    yolo = YoloUltralyticsClientZMQ(zmq_url="tcp://127.0.0.1:4000", width=640, height=640)
    img = cv2.imread("bus.jpg")
    yoloDets = yolo.detect(img)
    yoloTracks = yolo.track(yoloDets)
    print(yoloTracks, yoloDets)
