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
        with self.lock:
            if self.alert_active:
                return  # Avoid duplicate alerts
            if not filename.lower().endswith(".mp4"):
                filename = filename + ".mp4"
            # To avoid absolute path when the save_dir is available
            if self.save_dir and filename[0] == "/":
                filename = filename[1:]
            filepath = os.path.join(self.save_dir, f"{filename}")
            # Estimate FPS dynamically from recent timestamps
            estimated_fps = self._get_estimated_fps()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filepath, fourcc, estimated_fps, (self.frame_width, self.frame_height))
            
            print(f"Alert triggered! Recording video: {filepath} (FPS: {estimated_fps:.2f})")

            # Write buffered frames from the last `buffer_seconds` (before alert)
            for ts, frame in self.frame_buffer:
                self.video_writer.write(frame)

            self.alert_active = True
            self.alert_end_time = timestamp + self.record_seconds  # Set stop time
            self.filepath = filepath

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