import numpy as np
import ffmpeg
import cv2
cv2.setNumThreads(1)
import os

class OutStream:
    def __init__(self, out_xy, output_path):
        self.out_xy = out_xy
        print(self.out_xy)
        self.process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_xy[0]}x{out_xy[1]}')
            .output(output_path)
            # .output(output_path, format='ffm')  # Explicitly set format
            .overwrite_output()
            .global_args('-loglevel', 'error')
            .run_async(pipe_stdin=True)
        )

    def write(self, frame):
        frame = cv2.resize(frame, self.out_xy, interpolation=cv2.INTER_NEAREST)
        self.process.stdin.write(frame.tobytes())

import atexit
import signal
import sys


class OutStreamSaveVideo:
    def __init__(self, out_xy, output_path, fps=30, codec="mp4v"):
        self.out_xy = out_xy
        self.fps = fps
        self.output_path = output_path

        # Define codec and create VideoWriter object
        # fourcc = cv2.VideoWriter_fourcc(*codec)  # "mp4v" for MP4 format
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, out_xy)

        # Register cleanup functions
        atexit.register(self.close)
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)

    def write(self, frame):
        """Write a frame to the video stream."""
        frame = cv2.resize(frame, self.out_xy, interpolation=cv2.INTER_NEAREST)
        self.writer.write(frame)

    def close(self):
        """Release the video writer."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def _handle_exit(self, signum, frame):
        """Handle forced exits (Ctrl+C, kill command)."""
        print("\nInterrupt received! Saving video...")
        self.close()
        sys.exit(0)

    def __del__(self):
        """Ensure cleanup when the object is deleted."""
        self.close()


# # 1. Output Saver
# class VideoOut:
#     def __init__(self, output_path, fps=10, width=640, height=340, save_flag=True):
#         self.save_flag = save_flag
#         if self.save_flag:
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
#             self.out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
#             self.width = width
#             self.height = height
        
#     def save_frame(self, frame):
#         if self.save_flag:
#             frame = cv2.resize(frame, (self.width, self.height))
#             self.out.write(frame)

#     def release(self):
#         if self.save_flag:
#             self.out.release()

class SaveImg:
    def __init__(self, folder_out="./out_img"):
        os.makedirs(folder_out, exist_ok=True)
        self.i = 0 
        self.folder_out = folder_out
        self.width, self.height = 640, 480
        
    def save_img(self, frame):
        frame = cv2.resize(frame, (self.width, self.height))
        path_out = f"{self.folder_out}/img_{self.i}.jpg"
        cv2.imwrite(path_out, frame)
        print(path_out)
        self.i += 1