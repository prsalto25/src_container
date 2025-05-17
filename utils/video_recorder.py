import cv2
cv2.setNumThreads(1)
import os
import time
import threading
import numpy as np
from collections import deque
from pathlib import Path
import shutil
import subprocess


latest_frame = None
latest_lock = threading.Lock()

def set_latest_frame(frame):
    global latest_frame
    with latest_lock:
        latest_frame = frame.copy()

def get_latest_frame():
    global latest_frame
    with latest_lock:
        if latest_frame is not None:
            return True, latest_frame.copy()
        else:
            return False, None


class VideoRecorder:
    def __init__(self, save_dir="", buffer_seconds=3, record_seconds=5, frame_size=(640, 360)):
        self.save_dir = save_dir
        self.buffer_seconds = buffer_seconds  # Keep frames for last N seconds
        self.record_seconds = record_seconds  # Record additional seconds after alert
        self.frame_width = frame_size[0]
        self.frame_height = frame_size[1]
        self.frame_buffer = deque()  # Stores (timestamp, frame) tuples
        self.alert_active = False
        self.video_writer = None
        self.alert_end_time = None
        self.lock = threading.Lock()  # Ensures thread safety
        self.fps_estimates = deque(maxlen=10)  # Stores recent FPS estimates
        self.filepath = os.path.join(self.save_dir, "unknown.mp4")

        # Ensure save directory exists
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    def process_frame(self, frame, timestamp):
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))  # Ensure uniform size
        set_latest_frame(frame)
        # timestamp = time.time()  # Get current time

        # Calculate FPS dynamically based on timestamp differences
        if len(self.frame_buffer) > 1:
            last_timestamp = self.frame_buffer[-1][0]
            frame_interval = timestamp - last_timestamp
            if frame_interval > 0:
                current_fps = 1.0 / frame_interval
                self.fps_estimates.append(current_fps)  # Store FPS estimate

        self.frame_buffer.append((timestamp, frame))  # Store frame with timestamp

        # Remove old frames (older than buffer_seconds)
        while self.frame_buffer and (timestamp - self.frame_buffer[0][0]) > self.buffer_seconds:
            self.frame_buffer.popleft()

        with self.lock:
            if self.alert_active:
                self.video_writer.write(frame)
                if timestamp >= self.alert_end_time:
                    self._stop_recording()  # Stop recording when time is up

    def trigger_alert(self, filename, timestamp):
        if not filename.lower().endswith(".mp4"):
            filename = filename + ".mp4"
        if self.save_dir and filename[0] == "/":
            filename = filename[1:]

        filepath = os.path.join(self.save_dir, filename)

        with self.lock:
            # Make a copy of the buffer for this alert
            buffer_copy = list(self.frame_buffer)

        # Estimate FPS dynamically
        estimated_fps = self._get_estimated_fps()

        # Start a new recording thread
        alert_thread = RecordingThread(
            filepath=filepath,
            buffer_copy=buffer_copy,
            frame_width=self.frame_width,
            frame_height=self.frame_height,
            estimated_fps=estimated_fps,
            record_seconds=self.record_seconds
        )
        alert_thread.start()
        print(f"Parallel alert started: {filepath}")

    def _stop_recording(self):
        """
        Internal method to stop recording and release the video writer.
        """
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            time.sleep(0.5)  # make sure file is already written
            # Run file processing in a separate thread
            threading.Thread(target=self._process_saved_video, args=(self.filepath,), daemon=True).start()
        
        self.alert_active = False

    def _process_saved_video(self, filepath):
        """
        Handles the file copying and compression in a separate thread.
        """
        copy_filepath = filepath + ".copy"
        shutil.copy(filepath, copy_filepath)
        subprocess.run(f"ffmpeg -i {copy_filepath} -vcodec libx264 {filepath} -y", shell=True)
        
        # Remove the temporary copy after processing
        try:
            os.remove(copy_filepath)
            print(f"Temporary file {copy_filepath} removed.")
        except Exception as e:
            print(f"Error removing temporary file {copy_filepath}: {e}")

        print("Alert recording saved successfully.")

        
    def _get_estimated_fps(self):
        if len(self.fps_estimates) < 2:
            return 15  # Default fallback FPS
        return np.mean(self.fps_estimates)  # Use the average recent FPS
    

class RecordingThread(threading.Thread):
    def __init__(self, filepath, buffer_copy, frame_width, frame_height, estimated_fps, record_seconds):
        super().__init__(daemon=True)
        self.filepath = filepath
        self.buffer_copy = buffer_copy
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.estimated_fps = estimated_fps
        self.record_seconds = record_seconds

    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(self.filepath, fourcc, self.estimated_fps, (self.frame_width, self.frame_height))

        print(f"[RecordingThread] Writing pre-alert buffer: {len(self.buffer_copy)} frames")
        for ts, frame in self.buffer_copy:
            video_writer.write(frame)

        start_time = time.time()
        while time.time() - start_time < self.record_seconds:
            ret, frame = get_latest_frame()
            if ret:
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                video_writer.write(frame)
            time.sleep(1.0 / self.estimated_fps)

        video_writer.release()

        # Optional compression
        try:
            copy_path = self.filepath + ".copy"
            shutil.copy(self.filepath, copy_path)
            subprocess.run(f"ffmpeg -i {copy_path} -vcodec libx264 {self.filepath} -y", shell=True)
            os.remove(copy_path)
            print(f"[RecordingThread] Finished and compressed: {self.filepath}")
        except Exception as e:
            print(f"[RecordingThread] Compression error: {e}")
