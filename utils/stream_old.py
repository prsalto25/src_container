from threading import Thread
import queue
import sys
import numpy as np
import cv2
import time
from collections import deque

class FileVideoStream_live:
    def __init__(self, path, screenXY):
        self.stream = cv2.VideoCapture(path)
        self.path = path
        self.q = deque([], maxlen=1)
        self.x = screenXY[0]
        self.y = screenXY[1]
        self.len = 1

        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()

    def update(self):
        while True:
            try:
                ret, frame = self.stream.read()
                time.sleep(.04)
                if frame is not None:
                    frame = cv2.resize(frame, (self.x, self.y))
                    self.q.append(frame)
                    continue
                else:
                    pass
                    #print('stream.read error1')
            except:
                print('stream.read error2')

            # re-capture video
            time.sleep(.1)
            try:
                self.stream.release()
                self.stream = cv2.VideoCapture(self.path)
            except:
                print('stream.read error3')

    def read(self):
        while True:
            try:
                return self.q.pop()
            except:
                time.sleep(.1)

    def release(self):
        self.stream.release()
    
class FileVideoStream_video:
    def __init__(self, path, screenXY, restart, wait, skip):
        self.stream = cv2.VideoCapture(path)
        self.path = path
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        self.Q = queue.Queue(maxsize=5)
        self.x = screenXY[0]
        self.y = screenXY[1]
        self.len = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.restart = restart
        self.last_frame_time = time.time()
        self.wait = wait
        self.finish = False
        self.skip = skip

        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()

    def update(self):
        while True:
            for i in range(self.skip):
                ret, frame = self.stream.read()
            #time.sleep(.05)
            if frame is not None:
                frame = cv2.resize(frame, (self.x, self.y))
                wait = self.wait - (time.time() - self.last_frame_time) # set maximum fps = 25
                if wait > 0:
                    time.sleep(wait)
                self.Q.put(frame)
                self.last_frame_time = time.time()
            elif self.restart:
                self.stream.release()
                self.stream = cv2.VideoCapture(self.path)
            else:
                self.Q.put(None)
                self.finish = True
                break

    def read(self):
        if self.finish:
            return None
        else:
            return self.Q.get()

    def release(self):
        self.stream.release()

def createFileVideoStream(mode, path, screenXY, restart, wait=.05, skip=0):
    if mode == 'auto' and (path[:4] == 'rtsp' or path[:4] == 'http'):
        return FileVideoStream_live(path, screenXY)
    elif mode == 'live':
        return FileVideoStream_live(path, screenXY)
    else:
        return FileVideoStream_video(path, screenXY, restart, wait, skip)


if __name__ == "__main__":
    #import sys
    from tqdm import tqdm

    screenXY = (1280,720)

    path = "rtsp://admin:Admin123!@192.168.31.198:554/onvif/profile1/media.smp"
    fvs = FileVideoStream(path, 'live', screenXY)
    fvs.start()
    pbar = tqdm(total=fvs.len)
    while 1:
        pbar.update(1)
        time.sleep(1)
        frame = fvs.read()
        if frame is None:
            break
        print(frame.shape)
        # -------display------
        #cv2.imshow('img', frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break
        #--------video--------
    pbar.close()
    fvs.release()
    cv2.destroyAllWindows()

