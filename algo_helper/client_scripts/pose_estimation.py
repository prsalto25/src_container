import numpy as np
import zmq
import cv2
cv2.setNumThreads(1)

import socket
def _check_port_active(host: str, port: int) -> bool:
    """Check if a port is active on a given host.
    - ex usage: 
    _flag_server_is_on = _check_port_active("localhost", int("2687"))
    print(f"Pose Estimator server is active : {_flag_server_is_on}")
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2)  # Set a timeout for faster response
        result = s.connect_ex((host, port))
        return result == 0  # Returns True if the port is open


class PoseEstimator:
    def __init__(self, docker_url='tcp://127.0.0.1:2687'):
        #2557
        _flag_server_is_on = _check_port_active("localhost", int("2687"))
        print(f"Pose Estimator server is active : {_flag_server_is_on}")
        
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(docker_url)
        

    def detect(self, frame, bboxes):
        # 1. Send to pose
        frame = frame.astype(np.uint8)
        frame_shape = frame.shape
        
        bboxes = bboxes.reshape(-1, 4).astype(int)

        text_frame_shape = f"{frame_shape[0]}_{frame_shape[1]}_{frame_shape[2]}"

        self.socket.send_multipart([frame, text_frame_shape.encode(), bboxes])

        # 2. Recv from pose
        pose_result, scores = self.socket.recv_multipart()
        pose_result = np.frombuffer(pose_result, dtype=np.float32)
        frame = cv2.rectangle(frame, bboxes[0, :2], bboxes[0, 2:], (255,0,0), 2)
        
        if pose_result.size:
            pose_result = pose_result.reshape(-1, 17, 2)
            scores = np.frombuffer(scores, dtype=np.float32).reshape(-1, 17, 1)

            # Keep only highest confident pose
            # pose_result = pose_result[np.argmax(scores)]
        return np.concatenate((pose_result, scores), axis=2)