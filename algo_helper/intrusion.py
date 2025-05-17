# old: http://git.graymatics.com/kelvin/algo_helper_spam/src/branch/master/Intrusion.py
import os
import re
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
import time
import cv2
cv2.setNumThreads(1)

from .use import drawing
from .use.point_in_poly import point_in_poly
from .use.duration_object import DurationManager


class Intrusion:
    def __init__(self, algos_dict, outstream, mysql_helper):
        self.attr = algos_dict
        self.outstream = outstream
        self.duration_manager = DurationManager(expiration_threshold=20)

    def _trigger_alert(self, frame, tid):
        # date = datetime.now().strftime("%Y-%-m-%-d_%H:%M:%S")
        # id_ = str(uuid.uuid4())
    
        # # Set to None if you dont want to send anything
        # image_fname = f"{date}_{tid}_zone0.jpg" 
        # video_fname = f"{date}_{tid}_video.mp4" 
        # _img_full_path = os.path.join(self.alert_folder_resource_path, image_fname)
        # alert_data = [date, self.attr['camera_name'], 0, tid, self.attr['camera_id'], self.attr['id_account'], self.attr['id_branch'], id_, 1]  # pcount=1 
        # tickets_data = (id_, 'Intrusion Alert', date, date, 'NULL', self.attr['id_account'], self.attr['id_branch'], 1, self.attr['camera_name'], 'NULL', _img_full_path,'NULL', self.attr['camera_id'])
        
        # self.save_alert(frame, image_fname, video_fname, alert_data, tickets_data)
        pass

    def run(self, frame, yolo_person):

        # Draw ROI
        if (self.attr['rois'] is not None):
            drawing.draw_rois(frame, self.attr['rois'], (255,0,0))

        # Checking Intrusion
        pcount = 0
        for i, track in enumerate(yolo_person):
            (x1,y1,x2,y2), (x,y,w,h), tid = track.attr['xyxy'], track.attr['xywh'], str(track.id)


            if w > 1000:
                continue
            if point_in_poly(x, y, self.attr['rois']):
                pcount += 1

                # Get Duration from trackid
                cur_duration = self.duration_manager.update(tid)

                text = f"{tid}: {cur_duration:.2f}s"
                drawing.plot_one_box(frame, (x1,y1), (x2, y2), color = (0, 0, 255), label=text, line_thickness=2, line_tracking=False, polypoints=None)
                drawing.putTexts(frame, ["INTRUSION DETECTED!!"], 500, 50, size=1.5, thick=2, color=(0,0,255))   
            else:
                drawing.plot_one_box(frame, (x1,y1), (x2, y2), color = (125, 125, 125), label=tid, line_thickness=2, line_tracking=False, polypoints=None)
                

       
        
        self.outstream.write(frame)
        # self.update_video_alert(frame)
        return frame















