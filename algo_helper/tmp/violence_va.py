import zmq
import ast
import numpy as np
import uuid
import cv2
import os
import copy
cv2.setNumThreads(1)


import math
import time
from datetime import datetime,timedelta
from collections import deque


import threading

from lru import LRU
import sys


sys.path.append('./algo_helper')
from .use import drawing as drawing
from .use.point_in_poly import point_in_poly
from .use.postprocessing_collapse import StateTracker
from .use.preprocessing import generate_pairs, is_duplicate
from .use.alert_video import AlertVideo ## FOR VIDEO ALERT
from .use.hit_duration import EventTimer #J

from .client_scripts.violence_client import ViolenceClsVLMKelvin
from .client_scripts.pose_estimation import PoseEstimator
from .client_scripts.violence_cls import ActionClassifier



d = datetime.now()
prev_time = d.strftime('%Y-%-m-%-d_%H:%M:%S')

class ViolenceVA:
    def __init__(self):
        self.lock = threading.Lock()  # Ensures thread safety
        self.vio_cls_vlm = ViolenceClsVLMKelvin()  # Initialize the violence classifier

    #USAGE ON OUR VA ! 
    def run_alert_in_thread(self, frame, confidence):
        alert_thread = threading.Thread(target=self.send_alert, args=(frame, confidence))
        alert_thread.start()

    def send_alert(self, frame, confidence): # need to write
        global prev_time
        print("going here? ")
        date = self.timer.now.strftime('%Y-%m-%d %H:%M:%S')

        ## ----------------------------------------------------------------------------------------
        ## MODIFICATION IS HERE !!!!!!!!!!!!!!
        # try:
        #     response_vio = self.vio_cls_vlm.classify_vio(frame)
        #     if response_vio["flag_violence"] == False:
        #         # return 
        #         pass

        #     print(response_vio)
        # except: 
        #     print("HAVE INTERNAL ERROR !!")
        # self.lock = threading.Lock()  # Ensures thread safety

        ## debug locally
        # cv2.imwrite(f"./_tmp_img/{str(uuid.uuid4())}.png", frame)
        # print('finish writing')




class Violence(ViolenceVA):
    def __init__(self, timer, outstream, mysql_helper, algos_dict, elastic):
        super().__init__()
        self.outstream = outstream
        self.attr = algos_dict
        self.timer = timer
        self.es = elastic 
        self.castSize = (640, 480)

        mysql_fields = [
            ['time','datetime'],
            ['clip_path','varchar(45)'],
            ["confidence","float"],
            ["camera_name","varchar(45)"],
            ["cam_id","varchar(45)"],
            ['id_branch',"varchar(45)"],
            ["id_account","varchar(45)"],
            ['id',"varchar(45)"],
            ['severity',"varchar(45)"]
        ]
        self.tcount = 0
        self.last_sent = time.time()
        self.alertVideo = AlertVideo()        
        self.count = 0

        # Actions Estimate.
        self.action_model = ActionClassifier()
        self.tracker_management = {}
        self.roi_checker = None

        ## State dict
        self.state_dict = LRU(50)
        self.hits = dict()
        self.tracked = []

        ## SIZE BBOX (max_h, max_w = 540, 960)
        ## PARAM FOR COLLAPSE
        self.without_edge_pp = False
        # analytics_param = algos_dict["analytics_param"]
        analytics_param = False
        if analytics_param:
            self.th0_w = analytics_param.th0_w
            self.th1_w = analytics_param.th1_w
            self.th0_h = analytics_param.th0_h
            self.th1_h = analytics_param.th1_h
            self.th0_area = analytics_param.th0_area
            self.th1_area = analytics_param.th1_area
            self.alert_thk_th = analytics_param.alert_thk_th_collapse # 30s 60s
            self.th_n_roi_points = analytics_param.th_n_roi_points
            self.th_collapse_prob = analytics_param.th_collapse_prob
            try:
                self.without_edge_pp = analytics_param.without_edge_pp
                print("WITHOUT EDGE PP RULES !!!!!!!!!!!!!!!!")
            except:
                self.without_edge_pp = False
            print(vars(analytics_param))
        else:
            self.th0_w = 30
            self.th1_w = 600
            self.th0_h = 50
            self.th1_h = 500
            self.th0_area = 30*30
            self.th1_area = 600*150
            self.alert_thk_th = 60*5 # 30s 60s
            self.th_n_roi_points = 7
            self.th_collapse_prob = 0.7

        ## PARAM FOR VIOLENCE
        analytics_param_violence = False
        if analytics_param_violence:
            self.n_event_hit_violence = analytics_param_violence.n_event_hit_violence
            self.scale_up_h = analytics_param_violence.scale_up_h
            self.scale_up_w = analytics_param_violence.scale_up_w
            self.th_conf_violence = analytics_param_violence.th_conf_violence
            self.th_people_dist = analytics_param_violence.th_people_dist
            self.len_deque_violence_seq = analytics_param_violence.len_deque_violence_seq
            self.min_move_speed = analytics_param_violence.min_move_speed
        else:
            self.n_event_hit_violence = 8 #5
            self.scale_up_h = 0.1
            self.scale_up_w = 0.25
            self.th_conf_violence = 0.85
            self.th_people_dist = 500
            self.len_deque_violence_seq = 5*10
            self.min_move_speed = 4

        self.fs_labels = ["hair_pulling", "handshake", "kicking", "punching", "purse_snatching", "pushing", "talking", "walking"]
        self.tcount = 0
        self.castSize = (640, 480)
        
        #DEBUG
        self.alert_counts = 0
        self.cooldown = 120/20
     
        self.time_since_alert = time.time() - self.cooldown
        self.alert_flag =True
        self.lock = threading.Lock()  # Ensures thread safety
        self.vio_cls_vlm = ViolenceClsVLMKelvin()  # Initialize the violence classifier

        self.hit_duration = EventTimer(time_expired=0.5)  #J

        # from utils_infer.stream_out import SaveImg
        # self.img_saver = SaveImg("./img_out_6")


    
    def check_near_fov(self, x1, y1, x2, y2, frame_shape):
        h, w, _ = frame_shape
        # Only Edge of the Bottom, right and left (only cropped rect that will have same value as max of the frame)
        if x1 == 0 or x2 >= (w-5) or y2 >= (h-5):
            return True
        else:
            return False
    
    def send_video(self, frame, date, id_):
        date = self.timer.now.strftime('%Y-%-m-%d %H:%M:%S')
        #imgName=f"/resources/{self.attr['id_account']}/{self.attr['id_branch']}/violence/{self.attr['camera_id']}/{date}_{id_}.jpg"
        videoName = f"/resources/{self.attr['id_account']}/{self.attr['id_branch']}/violence/{self.attr['camera_id']}/{date}_{id_}.mp4"
        videoPath = '/home' + videoName
        resizedFrame = cv2.resize(frame, (640,360))
        self.alertVideo.trigger_alert(videoPath,videoName,10) ## INCREASE OR DECREASE THE DURATION                                           
        drawing.saveVideo(videoPath, resizedFrame)
        return videoName

    ## FOR IMAGE ALERT
    def send_img(self, frame, date, id_):
        date = self.timer.now.strftime('%Y-%m-%d %H:%M:%S')
        imgName = f"/resources/{self.attr['id_account']}/{self.attr['id_branch']}/violence/{self.attr['camera_id']}/{date}_{id_}.jpg"
        imgPath = '/home' + imgName
        resizedFrame = cv2.resize(frame, self.castSize)
        drawing.saveImg(imgPath, resizedFrame)
        return imgName


    def run(self, frame, yolo_track_person):
        # print('violence ',arr_person_bbox)
        arr_person_bbox = [] 
        for i, track in enumerate(yolo_track_person):
            x1, y1, x2, y2 = track.attr['xyxy']
            texts = [track.id[-3:]]
            tid = int(texts[-1].replace('_',''))
            arr_person_bbox.append([x1, y1, x2, y2, 1, tid])
        arr_dets = np.array(arr_person_bbox).astype(np.int)

      
        ts = time.time()  #J
        frame_original = frame.copy()

        if not arr_dets.size:
            # nothing detected
            self.outstream.write(frame)
            return
        frame_copy = frame.copy()
      
        for i, det in enumerate(arr_dets):
            if self.state_dict.get(det[5]):
                state = self.state_dict[det[5]]
                state.update_state(det[:4])
            else:
                state = self.state_dict[det[5]] = StateTracker(det[5])
                state.update_state(det[:4])

       
        filtered_dets = []
        for i, det in enumerate(arr_dets):
            x1, y1, x2, y2, conf, tid = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = int(abs(x2 - x1)), int(abs(y2 - y1))
            
            if not point_in_poly(x1, y1, self.attr['rois']):
                continue

            if not point_in_poly((x2-(w/2)), y2, self.attr['rois']):
                continue
            
            # Size filter
            if w < 64 and h < 64:
                continue

            # ## b. bbox near edge of frame
            if not self.without_edge_pp:
                if self.check_near_fov(x1, y1, x2, y2, frame.shape):
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(125,125,0),2)
                    continue

 

            bbox = [x1, y1, x2, y2, conf, tid]
            filtered_dets.append(bbox)

        filtered_dets = np.array(filtered_dets)

        if filtered_dets.size:
            filtered_dets = filtered_dets.reshape(-1, 6)
        else:
            self.outstream.write(frame)
            return

        ## (FOR VIOLENCE: collect centroid coordinates)
        centroid_list = []
        for tid in filtered_dets[:, 5]:
            if self.state_dict[tid].centroid_buffer:
                centroid_list += [self.state_dict[tid].centroid_buffer[-1]]
        centroid_list = np.array(centroid_list)

        bbox_pairs = []
        mask = [False for i in range(len(filtered_dets))]

        # We store the the position of the bbox in the filtered dets
        # For better performance
        tid_pairs = []
        for i, det in enumerate(filtered_dets):
            x1, x2, tid = det[[0, 2, 5]]

            # Get centroid of current box and its dist with other boxes
            centroid = np.array(self.state_dict[tid].centroid_buffer[-1])
            dist_centroids = np.linalg.norm(centroid_list - centroid, axis=1)

            normed_dists = dist_centroids / (x2 - x1)
            valid_dists = [dist < 1.00 and dist > 0.01 for dist in normed_dists]
            valid_bboxes = filtered_dets[valid_dists]
            valid_tids = filtered_dets[valid_dists, 5]

            # Boolean OR the mask
            mask = [a or b for a, b in zip(valid_dists, mask)]

            if valid_bboxes.size:
                bbox_pairs.append([det, valid_bboxes])
                tid_pairs.append([tid, valid_tids])

                # Also include the current bbox
                mask[i] = True
        
        # Obtain all the boxes
        all_boxes = filtered_dets[mask]
    
              # ANALYTICS: Pose Detection and Classification
           
        stage_2_class_results  = self.action_model.detect_single(
            frame_copy, bbox_pairs, tid_pairs, all_boxes
        )

        # PP: Pose movement - update pose buffer
        pose_results_dict = dict()
        for _, _, pose_pair, bbox_pair_meta in stage_2_class_results:
            for pose, bbox in zip(pose_pair, bbox_pair_meta[0]):
                pose_results_dict[bbox[5]] = pose

        for i, result in enumerate(stage_2_class_results):
            # Extract values
            class_result, pose_results, pose_pair, bbox_pair = result
            
            # Get merged bbox and tid pair
            x1, y1, x2, y2 = bbox_pair[1]
            tid_pair = tuple(frozenset(np.array(bbox_pair[0])[:, 5]))

            # Get best class name, score and draw
            class_name, prob = self.action_model.get_best_class(class_result)
            txt = class_name
      
            
            class_id = 0
            vio_duration = 0
        
            if (class_name == "violence" and prob > 0.50):
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # drawing.plot_one_box(frame, (x1,y1), (x2, y2), color = (0, 0, 255), label=f"violence", line_thickness=3, line_tracking=False, polypoints=None)
                vio_duration = self.hit_duration.hit()
                drawing.plot_one_box(frame, (x1,y1), (x2, y2), color = (0, 0, 255), label=f"violence: {vio_duration:.0f}s", line_thickness=3, line_tracking=False, polypoints=None)  #J              
                # drawing.plot_one_box(frame, (x1,y1), (x2, y2), color = (0, 0, 255), label=f"violence: {vio_duration}s", line_thickness=3, line_tracking=False, polypoints=None)  #J              
                
                
                # print("glimpse of violence")  #J
                class_id = 1
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                class_id = 0

            ## Collect Class
            cur_track_hits = self.hits.get(tid_pair, deque(maxlen=self.len_deque_violence_seq))
            cur_track_hits.append(class_id)
            self.hits[tid_pair] = cur_track_hits
            
            # print(tid_pair, self.hits[tid_pair])  #J
            cur_time = time.time() 
            if (sum(self.hits[tid_pair]) > self.n_event_hit_violence):
                """
                # self.flag_testing = True
                # self.hits[tid_pair].clear()
                #drawing.putTexts(frame, ['Violence'], 50, 50, size=2, thick=2, color=(0,0,255))
                if tid_pair not in self.tracked and cur_time - self.time_since_alert > self.cooldown:
                    print("VIOLENCE DETECTED !!!!!!!")
                    self.run_alert_in_thread(frame_original,frame, prob)
                    print(frame.shape)
                    print("RUN ALERT IN THREAD IS BEEN CALLED !!!!!!!!")
                    # self.send_alert(frame_original, frame, prob)
                    self.tracked.append(tid_pair)
                    self.time_since_alert = time.time()
                """  #J
                # if vio_duration > 5/20: # put this on self variable later
                if True:
                    # self.flag_testing = True
                    # self.hits[tid_pair].clear()
                    #drawing.putTexts(frame, ['Violence'], 50, 50, size=2, thick=2, color=(0,0,255))
                    # if tid_pair not in self.tracked and cur_time - self.time_since_alert > self.cooldown:
                    if True:
                        print("VIOLENCE DETECTED !!!!!!!")
                        # self.run_alert_in_thread(frame_original,frame, prob)
                        self.run_alert_in_thread(frame, 1)
                        print(frame.shape)
                        print("RUN ALERT IN THREAD IS BEEN CALLED !!!!!!!!")
                        # self.send_alert(frame_original, frame, prob)
                        self.tracked.append(tid_pair)
                        self.time_since_alert = time.time()
            elif cur_time - self.time_since_alert < self.cooldown:
               
                continue
      
        self.alertVideo.updateVideo(frame)
        # self.video_recorder.process_frame(frame, ts)  #J
        #print("STREAMING STARTS NOW !!!!!!!!!!!!!!!!!!")
        # print("send frames")
        # time.sleep(0.1)
        self.outstream.write(frame)
        self.count+=1   
        print(self.count)

        # self.img_saver.save_img(frame)
        

        # self.run_alert_in_thread(frame_original, 1)
        # print('going here')
        # time.sleep(1000)       