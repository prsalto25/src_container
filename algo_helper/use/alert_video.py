import cv2
import os
import ffmpeg
import time
import requests
from pathlib import Path
import shutil
import subprocess

class VideoWriter:
    def __init__(self, filename,duration):
        self.frame_shape = (640,360)
        self.writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 15, self.frame_shape)
        self.duration = duration
        self.start_time = time.time()
        self.filename = filename

    def write(self, frame):
        if time.time() - self.start_time > self.duration: 
            self.start_time = time.time()
            self.writer.release()
            print('sgd', self.filename)
            shutil.copy(self.filename, self.filename + '.copy')
            subprocess.run(f"ffmpeg -i {self.filename}.copy -vcodec libx264 {self.filename} -y", shell = True)
            return True
        else:
            frame = cv2.resize(frame, self.frame_shape)
            self.writer.write(frame)
        
            return False

class AlertVideo:
    def __init__(self):
        self.alerts = []

    def updateVideo(self, frame):
        alerts_incomplete = []
        for alert in self.alerts:
            isComplete = alert['videoWriter'].write(frame)
            if not isComplete:
                alerts_incomplete.append(alert)
        self.alerts = alerts_incomplete
        


    def trigger_alert(self, dir_, filename,duration, data={}):
        dir_base = '/home' # API
        now = time.time()

        Path(f"{os.path.dirname(dir_)}").mkdir(parents=True, exist_ok=True) 
        alert = {
                'alert_type': dir_,
                'video_path': f'{dir_base}{filename}',
                'time': int(now),
                'frame':None,
                'data':data
                }
    
        alert['videoWriter'] = VideoWriter(alert['video_path'], 10)
        self.alerts.append(alert)
           

if __name__ == '__main__':
    print(get_token().content)
