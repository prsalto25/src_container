import cv2
import sys
import numpy as np
import tritonclient.grpc as grpcclient

class _YoloV8ClientOnly:
    """
    http://git.graymatics.com/kelvin/yolov8_all_triton/src/branch/master/yolov8_client_script/yolov8_client.py
    A client to interact with a YOLOv8 model deployed on Triton Inference Server.
    """
    def __init__(self, name="yolov8m", url="172.17.0.1:8001", width=640, height=640):
        # 1. Setup Triton Client
        self.input_height, self.input_width = height, width
        self.model_name = name
        self.triton_client = self._prepare_triton_client(url=url, model_name=name)
        input_info, output_info = self._get_input_output_info(name)
        ## a. Inputs
        self.inputs = []
        for name, shape in input_info:
            print(f"Input Name: {name}, Shape: {shape}")
            self.inputs.append(grpcclient.InferInput(name, [1, shape[1], shape[2], shape[3]], "FP32"))
        ## b. Outputs
        self.outputs = []
        for name, shape in output_info:
            print(f"Output Name: {name}, Shape: {shape}")
            self.outputs.append(grpcclient.InferRequestedOutput(name))

    def _detect_only(self, frame):
        # LETTER BOX STILL WRONG ! 
        input_image_buffer = self._preprocess(frame, [self.input_height, self.input_width], letter_box=False)
        self.inputs[0].set_data_from_numpy(input_image_buffer)
        results = self.triton_client.infer(model_name=self.model_name, inputs=self.inputs, outputs=self.outputs)
        detected_objects = self._yolov8_postprocess(results)
        return detected_objects



    ### ===================================================================================
    # internal utils (please change _yolov8_postprocess not yolov8)
    def _preprocess(self, image, input_shape, letter_box=True):
        self.img_h, self.img_w, _ = image.shape
        if letter_box:
            new_h, new_w = input_shape[0], input_shape[1]
            offset_h, offset_w = 0, 0

            if (new_w / self.img_w) <= (new_h / self.img_h):
                new_h = int(self.img_h * new_w / self.img_w)
                offset_h = (input_shape[0] - new_h) // 2
            else:
                new_w = int(self.img_w * new_h / self.img_h)
                offset_w = (input_shape[1] - new_w) // 2

            resized = cv2.resize(image, (new_w, new_h))
            img = np.full((input_shape[0], input_shape[1], 3), 127, dtype=np.uint8)
            img[offset_h:(offset_h+new_h), offset_w:(offset_w+new_w), :] = resized # Middle.
        else:
            img = cv2.resize(image, (input_shape[1], input_shape[0]))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img /= 255.0        
        input_image_buffer = np.expand_dims(img, axis=0)
        return input_image_buffer

    def _yolov8_postprocess(self, results, letter_box=True):
        num_dets = results.as_numpy("num_dets")
        det_boxes = results.as_numpy("bboxes")
        det_scores = results.as_numpy("scores")
        det_classes = results.as_numpy("labels")

        boxes = det_boxes[0, :num_dets[0][0]] / np.array([self.input_width, self.input_height, self.input_width, self.input_height], dtype=np.float32)
        scores = det_scores[0, :num_dets[0][0]] # Because the topk is included in the tensorrt engine.
        classes = det_classes[0, :num_dets[0][0]].astype(int)
        
        old_h, old_w = self.img_h, self.img_w
        offset_h, offset_w = 0, 0

        boxes = boxes * np.array([old_w, old_h, old_w, old_h], dtype=np.float32)
        if letter_box:
            boxes -= np.array([offset_w, offset_h, offset_w, offset_h], dtype=np.float32)
        boxes = boxes.astype(int) # [x1, y1, x2, y2]

        detected_objects = []
        for box, score, label in zip(boxes, scores, classes):
            detected_objects.append([label, score, box])
        return detected_objects
        
    def _get_input_output_info(self, model_name, model_version="1"):
        metadata = self.triton_client.get_model_metadata(model_name=model_name, model_version=model_version)
        config = self.triton_client.get_model_config(model_name=model_name, model_version=model_version)

        input_info = [(input_metadata.name, input_metadata.shape) for input_metadata in metadata.inputs]
        output_info = [(output_metadata.name, output_metadata.shape) for output_metadata in metadata.outputs]
        return input_info, output_info

    @staticmethod
    def _prepare_triton_client(url, model_name):
        try:
            triton_client = grpcclient.InferenceServerClient(
                url=url,
                verbose=False,
                ssl=False,                                     # Enable SSL encrypted channel to the server.
                root_certificates=False,                       # File holding PEM-encoded root certificates.
                private_key=None,                              # File holding PEM-encoded private key.
                certificate_chain=None)      # File holding PEM-encoded certicate chain.
        except Exception as e:
            print("context creation failed: " + str(e))
            sys.exit()

        # Health check.
        if not triton_client.is_server_live():
            print("FAILED: is_server_live")
            sys.exit(1)
        if not triton_client.is_server_ready():
            print("FAILED: is_server_ready")
            sys.exit(1)
        if not triton_client.is_model_ready(model_name):
            print("FAILED: is_model_ready")
            sys.exit(1)
        return triton_client



from tracking.tracker.byte_tracker_yolox import BYTETracker
class YoloV8ClientTracker(_YoloV8ClientOnly):
    """
    Extension of YoloV8Client with object tracking capabilities.
    """
    def __init__(self, name, url, width, height):
        super().__init__(name, url, width, height)
        self.names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        # Initialize trackers for selected classes
        self.class_groups = ['person', 'vehicle', 'bicycle', 'animal', 'object', 'bag']
        self.trackers = {
            cls: BYTETracker() for cls in self.class_groups
        }

    def detect(self, frame):
        """
        Main detection.
        """
        yoloDets = self._detect_only(frame)
        return self._format(yoloDets)

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

        for cls_id, conf, (x1, y1, x2, y2) in detections:
            conf *= 100
            class_name = self.names[int(cls_id)]

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
            elif class_name in ['backpack', 'suitcase', 'handbag'] and conf > 15:
                formatted_detections['object'].append(detection)
            elif class_name in ['backpack', 'handbag'] and conf > 15:
                formatted_detections['bag'].append(detection)

        return formatted_detections

if __name__ == "__main__":
    yolo = YoloV8ClientTracker(name="yolov8m", url="172.17.0.1:12001", width=640, height=640)
    img = cv2.imread("test.png")
    yoloDets = yolo.detect(img)
    yoloTracks = yolo.track(yoloDets)
    print(yoloTracks, yoloDets)