
names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

 
class Yolo:
    def __init__(self):
        ## Triton YoloV7
        self.input_height = 704#288
        self.input_width = 1280#512
        self.model_name = "yolov7_w6"#_512x288" # Should be the same as the folder name.
        self.triton_client, self.inputs, self.outputs = self.prepare_triton_yolov7(self.input_height, self.input_width, self.model_name)
        ## Tracker
        self.algoSize = (1280, 720)
        self.yoloTracker = {}
        for cls in ['person', 'object', 'vehicle', 'animal', 'bicycle','bag']:
            self.yoloTracker[cls] = BYTETracker() #bytetracker

    def detect(self, frame):
        # 1. Prepare Input
        self.input_image_buffer = self.yolov7_preprocess(frame, [self.input_height, self.input_width])
        self.inputs[0].set_data_from_numpy(self.input_image_buffer)
        # 2. Detect
        detections = self.triton_client.infer(model_name=self.model_name,
                                            inputs=self.inputs,
                                            outputs=self.outputs,
                                            client_timeout=None)
        # 3. Post Processing Output
        yoloDets = self.yolov7_postprocess(detections)
        yoloDets = self.format(yoloDets)
        return yoloDets

    @staticmethod
    def format(dets_all, mot=False):
        dets = {cls:[] for cls in {'person', 'vehicle', 'bicycle', 'animal', 'backpack', 'suitcase','handbag','object','bag'}}
        for cls, conf, [x1, y1, x2, y2] in dets_all:
            conf *= 100
            if mot:                                                             
                det = [int(x1), int(y1), int(x2), int(y2), conf, cls]                
            else:                         
                x, y = int((x1 + x2)/2), int((y1 + y2)/2)                                       
                h, w = int(y2 - y1), int(x2 - x1)                                           
                det = {
                    "conf": conf,
                    "cls": names[int(cls)],
                    "xyxy": [x1, y1, x2, y2],
                    "xywh": [x,y,w,h],
                }

            # append  
            if names[int(cls)] == 'person' and float(conf) > 35:#CI
                dets['person'].append(det)
            elif names[int(cls)] in ['car', 'motorcycle', 'bus', 'truck','airplane'] and float(conf) > 20: ## SOMETIMES FINE TUNING NEEDED
                dets['vehicle'].append(det)
            elif names[int(cls)] == 'bicycle' and float(conf) > 20:
                dets['bicycle'].append(det) 
            elif names[int(cls)] in {'cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe'} and float(conf) > 10:
                dets['animal'].append(det)
            elif names[int(cls)] in {'backpack','suitcase','handbag'} and float(conf) > 15: 
                dets['object'].append(det)
            elif names[int(cls)] in {'backpack','handbag'} and float(conf) > 15:  
                dets['bag'].append(det)      

        return dets

    def track(self, dets_all):
        tracks_all = {}
        for cls, tracker in self.yoloTracker.items():
            tracks = tracker.update(dets_all[cls])
            tracks_all[cls] = tracks
        return tracks_all

    def yolov7_preprocess(self, image, input_shape, letter_box=True):
        if letter_box:
            self.img_h, self.img_w, _ = image.shape
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

    def yolov7_postprocess(self, results, letter_box=True):
        num_dets = results.as_numpy("num_dets")
        det_boxes = results.as_numpy("det_boxes")
        det_scores = results.as_numpy("det_scores")
        det_classes = results.as_numpy("det_classes")

        boxes = det_boxes[0, :num_dets[0][0]] / np.array([self.input_width, self.input_height, self.input_width, self.input_height], dtype=np.float32)
        scores = det_scores[0, :num_dets[0][0]] # Because the topk is included in the tensorrt engine.
        classes = det_classes[0, :num_dets[0][0]].astype(np.int)

        old_h, old_w = self.img_h, self.img_w
        offset_h, offset_w = 0, 0

        boxes = boxes * np.array([old_w, old_h, old_w, old_h], dtype=np.float32)
        if letter_box:
            boxes -= np.array([offset_w, offset_h, offset_w, offset_h], dtype=np.float32)
        boxes = boxes.astype(np.int) # [x1, y1, x2, y2]

        detected_objects = []
        for box, score, label in zip(boxes, scores, classes):
            detected_objects.append([label, score, box])
        return detected_objects

    @staticmethod
    def prepare_triton_yolov7(input_height, input_width, model_name):
        INPUT_NAMES = ["images"]
        OUTPUT_NAMES = ["num_dets", "det_boxes", "det_scores", "det_classes"]
        # Create server context.
        try:
            triton_client = grpcclient.InferenceServerClient(
                url="172.17.0.1:12001",
                verbose=False,
                ssl=False,                                     # Enable SSL encrypted channel to the server.
                root_certificates=False,                       # File holding PEM-encoded root certificates.
                private_key=None,                              # File holding PEM-encoded private key.
                certificate_chain=None)                         # File holding PEM-encoded certicate chain.
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

        # Input Output Buffer
        inputs = []
        outputs = []
        inputs.append(grpcclient.InferInput(INPUT_NAMES[0], [1, 3, input_height, input_width], "FP32"))
        outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[0]))
        outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[1]))
        outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[2]))
        outputs.append(grpcclient.InferRequestedOutput(OUTPUT_NAMES[3]))
        return triton_client, inputs, outputs