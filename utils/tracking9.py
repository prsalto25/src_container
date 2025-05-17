import numpy as np
import time
#import cv2
#from scipy import optimize
from collections import deque

def make_cost_pos2(dets, tracks, thres):
    '''
    calculate cost matrix based on position difference
    pos_d: column; pos_t: row
    '''
    x0,y0,x1,y1 = [],[],[],[]
    for det in dets:
        x0.append(det['xywh'][0])
        y0.append(det['xyxy'][1])

    for track in tracks:
        x1.append(track.attr['xywh'][0])
        y1.append(track.attr['xyxy'][1])

    # convert to numpy array
    x0 = np.array(x0, dtype=np.float32)
    y0 = np.array(y0, dtype=np.float32)
    x1 = np.array(x1, dtype=np.float32)
    y1 = np.array(y1, dtype=np.float32)

    # compute abs diff between track and det position
    dx = np.abs(x0 - x1[:, np.newaxis])
    dy = np.abs(y0 - y1[:, np.newaxis])

    # compute cost
    cost = (dx**2 + dy**2)**.5
    cost[cost > thres] = 9999

    return cost


class Track:
    def __init__(self, det, id_, qsize):
        self.attr = det 
        self.id = id_
        self.miss = 0
        self.time0 = time.time()
        self.tag = set([])
        self.dict = {}
        self.faceid = deque([], maxlen=35)
        self.features = []
       

class Tracker:
    def __init__(self, screen_size, dist_thres=100, miss_thres=30):
        self.tracks = []
        self.id_now = 0
        self.miss = 0
        self.screen_size = screen_size
        self.id_base = str(int(time.time()))
        self.dist_thres = dist_thres
        self.miss_thres = miss_thres

    def add_new_track(self, det):
        self.id_now += 1
        self.id_show = '{}_{}'.format(self.id_base, self.id_now)
        self.tracks.append(Track(det, self.id_show, 20))

    def update_track(self, track_ind, det):
        self.tracks[track_ind].attr = det 
        self.tracks[track_ind].miss = 0
    
    def remove_track(self):
        tracks_temp = []
        for track in self.tracks:
            x,y,w,h = track.attr['xywh']
            if track.miss > self.miss_thres:
                continue
            if x < 0 or x> self.screen_size[0]:
                continue
            if y < 0 or y> self.screen_size[1]:
                continue
            tracks_temp.append(track)
        self.tracks = tracks_temp

    def match(self, scores_mat):
        ids_track = []
        ids_det = []
        for i in range(scores_mat.shape[1]): # in dets len
            min_score = scores_mat.min()
            if min_score >= 9999:
                break
            else:
                index = np.where(scores_mat == min_score)
                x, y = index[0][0], index[1][0] # ind of track, det
                scores_mat[x,:] = 9999
                scores_mat[:,y] = 9999
                ids_track.append(x)
                ids_det.append(y)
        return ids_track, ids_det

    def update(self, detections):
        # create cost, assignment
        if not any(self.tracks) or not any(detections): # no assignment if either dets/ tracks is empty
            assign_det, assign_track = [], []
        else:
            cost = make_cost_pos2(detections, self.tracks, self.dist_thres)
            assign_track, assign_det = self.match(cost.copy().astype(np.float32))

        # update tracks based on assignment
        for i, track_ind in enumerate(assign_track):
            det_ind = assign_det[i]
            self.update_track(track_ind, detections[det_ind])
            
        ## update missing track
        assign_track_set = set(assign_track)
        for i in range(len(self.tracks)):
            if i not in assign_track_set:
                self.tracks[i].miss += 1

        # add new track
        assign_det_set = set(assign_det)
        for i in range(len(detections)):
            if i not in assign_det_set:
                self.add_new_track(detections[i])

        # remove
        self.remove_track()

        #return self.tracks
        return [i for i in self.tracks if i.miss==0]


if __name__ == '__main__':
    tracker = Tracker()
    detections = [{'xywh':(1,2,3,4), 'xyxy':(1,2,3,4)}, {'xywh':(3,4,3,4), 'xyxy':(1,2,3,4)}]
    tracks = tracker.update(detections)
