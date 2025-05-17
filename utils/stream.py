import time
import cv2
cv2.setNumThreads(1)
from threading import Thread
from collections import deque

STREAMING_PREFIXES = (
    "rtsp://", "rtmp://", "http://", "https://", "ftp://", "webrtc://",
    "srt://", "hls://", "dash://", "mms://", "udp://", "tcp://", "rtp://"
)

def print_log(streams, log_interval=3):
    while True:
        for stream in streams:
            elapsed = time.time() - stream.start_time
            if elapsed > 0:
                fps = stream.frame_count / elapsed
                print(f"{stream.path}: {fps:.2f} FPS")
                stream.start_time = time.time()
                stream.frame_count = 0
        time.sleep(log_interval)

class LiveVideoStream:
    def __init__(self, path, max_fps=5):
        self.path = path
        self.max_fps = max_fps
        self.q = deque([], maxlen=1)
        self.last_frame_time = time.perf_counter()
        self.start_time = time.perf_counter()
        self.frame_count = 0
        self.stream = cv2.VideoCapture(path)
        
        t = Thread(target=self._update, daemon=True)
        t.start()

    def _update(self):
        while True:
            ret, frame = self.stream.read()
            if frame is not None:
                self.q.append(frame)
            else:
                time.sleep(0.1)

    def read(self):
        while not self.q:
            time.sleep(0.01)
        
        frame = self.q.pop()
        self.frame_count += 1
        target_time = self.last_frame_time + (1.0 / self.max_fps)
        
        while time.perf_counter() < target_time:
            time.sleep(0.01)
        
        self.last_frame_time = time.perf_counter()
        return frame
    
    def release(self):
        self.stream.release()

class FileVideoStream:
    def __init__(self, path, max_fps=5, skip_frame=5, restart=True):
        self.path = path
        self.max_fps = max_fps
        self.restart = restart
        self.start_time = time.perf_counter()
        self.frame_count = 0
        self.stream = cv2.VideoCapture(path)
        self.skip_frame = skip_frame
        self.last_read_time = time.perf_counter()

    def read(self):
        # Skip Frame Handler
        if self.skip_frame == 0:
            ret, frame = self.stream.read()
        else: 
            for _ in range(self.skip_frame):
                ret, frame = self.stream.read() 
            
        
        # Restart Handler
        if not ret:
            if self.restart:
                self.stream.release()
                self.stream = cv2.VideoCapture(self.path)
                ret, frame = self.stream.read()
            else:
                return None

        # FPS limit handler
        self.frame_count += 1
        target_time = self.last_read_time + (1.0 / self.max_fps)
        
        while time.perf_counter() < target_time:
            time.sleep(0.01)
        
        self.last_read_time = time.perf_counter()
        return frame
    
    def release(self):
        self.stream.release()

def createFileVideoStream(path, max_fps=5, log_interval=3, vid_skip_frame=0, vid_restart=True):
    streams = []
    if path.startswith(STREAMING_PREFIXES):
        stream = LiveVideoStream(path, max_fps)
    else:
        stream = FileVideoStream(path, max_fps, vid_skip_frame, vid_restart)
    streams.append(stream)
    
    t_log = Thread(target=print_log, args=(streams, log_interval), daemon=True)
    t_log.start()
    
    return stream

if __name__ == "__main__":
    video_path = "your_video.mp4"
    stream = createFileVideoStream(video_path, max_fps=5, log_interval=3, vid_restart=True)
    
    while True:
        frame = stream.read()
        if frame is None:
            break
        print("Frame received")
    
    stream.release()
