import cv2
cv2.setNumThreads(0)
from utils.tracking9 import Tracker
from tracking.tracker.byte_tracker_yolox import BYTETracker
import requests

class ANPR_Yolox:
    def __init__(self):
        self.anprTracker = {}
        for cls in ['Registration_Plate']:
            self.anprTracker[cls] = Tracker((1280, 720))
            # self.anprTracker[cls] = BYTETracker()

    def detect(self, frame):
        ret, im_np = cv2.imencode('.jpg', frame)
        im_byte = im_np.tobytes()
        dets = self.get_dets(im_byte)
        dets_reformatted = self.format_dets(dets)
        return dets_reformatted

    def get_dets(self, im_byte):
        #metadata = post('http://172.16.3.151:5004/predict', data=im_byte)
        metadata = requests.post('http://192.168.0.103:5004/predict', data=im_byte) ## IF THE NETWORK IS 'host'
        return metadata.json()

    def format_dets(self, raw_dets):
        detections = {cls:[] for cls in {'Registration_Plate'}}
        # boxes, scores, cls_ids, class_names = raw_dets
        # boxes = boxes.astype('int16').tolist()
        # cls_ids = cls_ids.astype('int16').tolist()
        # scores = scores.astype('float32').tolist()
        for det in raw_dets:
            x,y,w,h = det['xywh']
            x1,y1,x2,y2 = det['xyxy']
            conf = det['conf']
            _cls = det['cls']
            det = {}
            x, y = (x1+x2)/2, (y1+y2)/2
            w, h = x2-x1, y2-y1
            det['conf'] = conf
            det['cls'] = _cls
            det['xyxy'] = [x1,y1,x2,y2]
            det['xywh'] = [x,y,w,h]

            if _cls == 'Registration_Plate' and conf > .2:
                detections['Registration_Plate'].append(det)
        return detections

    def track(self, dets_all):
        # typeofdets = "anpr"
        tracks_all = {}
        for cls, tracker in self.anprTracker.items():
            tracks = tracker.update(dets_all[cls])
            tracks_all[cls] = tracks
        return tracks_all