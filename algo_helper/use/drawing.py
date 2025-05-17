#last edited - danny

import os
import cv2
from pathlib import Path
import random
import numpy as np

def saveImg(path, img):
    dir0 = os.path.dirname(path)
    Path(dir0).mkdir(parents=True, exist_ok=True)
    try:
        cv2.imwrite(path, img)
    except:
        print("failed to save image.")
        
def saveVideo(path, vdo):
    dir0 = os.path.dirname(path)
    Path(dir0).mkdir(parents=True, exist_ok=True)
    #video = VideoWriter(path, 5)
    #writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), 15, (640,360))
    #frame = cv2.resize(frame,(640,360))
    #writer.write(path,video)
    #isComplete = video.write(vdo,start_time)
    #return isComplete


class VideoWriter:
    def __init__(self, filename, duration):
        self.frame_shape = (640,360)
        self.writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 15, self.frame_shape)
        self.duration = duration
        self.start_time = time.time()

    def write(self, frame,start_time):
        if time.time() - start_time > 50:
            self.writer.release()
            return True
        else:
            #frame = cv2.resize(frame, self.frame_shape)
            self.writer.write(frame)
            return False

def draw_roi(frame, pts, color):
    for i in range(len(pts)-1):
        cv2.line(frame, tuple(pts[i]), tuple(pts[i+1]), color,2)
    cv2.line(frame, tuple(pts[-1]), tuple(pts[0]), color,2)

def draw_rois(frame, rois, color):
    for roi in rois:
        draw_roi(frame, roi, color)

def get_distance(p, q):
    return ((p[0]-q[0])**2 + (p[1]-q[1])**2)**.5

def putTextsS(img, texts, x1, y1, size=1, thick=1, color=(255,255,255)):
    for text1 in texts[::-1]:
        text1 = str(text1)
        (text_x, text_y) = cv2.getTextSize(text1, cv2.FONT_HERSHEY_PLAIN, fontScale=size, thickness=thick)[0]
        x1, y1 = int(x1), int(y1)
        cv2.rectangle(img, (x1,y1-text_y-10), ((x1+text_x+15), y1+15), (0, 0, 0), -1)
        cv2.putText(img, text1, (x1+10,y1), cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)
        y1 -= text_y
def putTexts(img, texts, x1, y1, size=1, thick=1, color=(255,255,255)):
    for text1 in texts[::-1]:
        text1 = str(text1)
        (text_x, text_y) = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, fontScale=size, thickness=thick)[0]
        x1, y1 = int(x1), int(y1)
        cv2.rectangle(img, (x1,y1-text_y-10), ((x1+text_x+15), y1+15), (0, 0, 0), -1)
        cv2.putText(img, text1, (x1+10,y1), cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)
        y1 -= text_y

def putTexts2(img, texts, x1, y1, size=1.5, thick=1, color=(0,0,0)):
    for text1 in texts[::-1]:
        text1 = str(text1)
        (text_x, text_y) = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, fontScale=size, thickness=thick)[0]
        x1, y1 = int(x1), int(y1)
        cv2.rectangle(img, (x1,y1-text_y-10), (x1+text_x+15, y1+15), (0, 0, 0), -1)
        cv2.putText(img, text1, (x1+10,y1), cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)
        y1 -= text_y * 1.2

def putTextCenter(img, texts, x1, y1, x2, size=1.5, thick=1, color=(0,0,0)):
    for text1 in texts[::-1]:
        text1 = str(text1)
        text_size = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, fontScale=size, thickness=thick)[0]

        x1, y1 = int(x1), int(y1)
        bbox_width = text_size[0] + 20
        bbox_height = text_size[1] + 25

        text_box_x = x1 + (x2 - x1 - bbox_width) // 2
        text_box_y = y1 - bbox_height

        cv2.rectangle(img, (text_box_x, text_box_y), (text_box_x + bbox_width, text_box_y + bbox_height), (0, 0, 0), -1)

        text_x = text_box_x + 10
        text_y = text_box_y + 20
        text_x = text_box_x + (bbox_width - text_size[0]) // 2 
        text_y = text_box_y + (bbox_height + text_size[1]) // 2 
        cv2.putText(img, text1, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)
        # y1 -= text_y * 1.2

def plot_one_box(img, c1, c2, color=None, label=None, line_thickness=3, line_tracking = None, polypoints=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    #c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    if line_tracking and polypoints:
        polylines(img, color, tl, polypoints)

def polylines(img,color=None, line_thickness=3, points = None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    cv2.polylines(img, np.array([points], np.int32), False, color, line_thickness)
