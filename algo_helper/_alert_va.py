import os
import pathlib
import threading
import time
import uuid
from datetime import datetime, timezone

import cv2
from utils.video_recorder import VideoRecorder



class AlertsVA:
    def __init__(
        self,
        algos_dict,
        mysql_helper,
        sql_table_name,
        sql_table_fields,
        alert_folder_resource_path=None,
        video_recorder_buffer_seconds=2,
        video_recorder_record_seconds=2,
        video_alert_resolution=(1280, 720),
        image_alert_resolution=(1280, 720),
    ):
        # Attr
        self.attr = algos_dict
        # Folder Alert (image & video)
        self.alert_folder_resource_path = alert_folder_resource_path
        pathlib.Path(self.alert_folder_resource_path).mkdir(parents=True, exist_ok=True)
        # Sql 
        self.mysql_helper = mysql_helper
        self.sql_table_name = sql_table_name
        if mysql_helper:
            self.mysql_helper.add_table(self.sql_table_name, sql_table_fields)

        # Img and Video Alert Setting
        self.video_alert_resolution = video_alert_resolution
        self.image_alert_resolution = image_alert_resolution

        self.video_recorder = VideoRecorder(
            buffer_seconds=video_recorder_buffer_seconds,
            record_seconds=video_recorder_record_seconds,
            frame_size=self.video_alert_resolution
        )


    def _send_video(self, frame, fname):
        video_path = os.path.join(self.alert_folder_resource_path, fname)
        self.video_recorder.trigger_alert(video_path, time.time())
        return video_path

    def _send_img(self, frame, fname):
        img_path = os.path.join(self.alert_folder_resource_path, fname)
        resized_frame = cv2.resize(frame, self.image_alert_resolution)
        cv2.imwrite(img_path, resized_frame)
        print(f"_send_img {img_path}")
        return img_path

    def _save_alert(self, frame, image_fname, video_fname, alert_data, tickets_data):
        print("Trigger Alert!")
        my_uuid = str(uuid.uuid4())
        my_date = datetime.now(timezone.utc).strftime("%Y-%-m-%-d_%H:%M:%S")

        # Send Image
        if image_fname: 
            self._send_img(frame, fname=image_fname)
        # Send Video
        # HACK: ffmpeg cannot save img due to resource issue
        # if video_fname:
            # self._send_video(frame, fname=video_fname)
        # Send SQL
        if alert_data:
            self.mysql_helper.insert_fast(self.sql_table_name, alert_data)
        if tickets_data:
            self.mysql_helper.insert_fast("tickets", tickets_data)
        if alert_data or tickets_data:
            self.mysql_helper.commit_all()
        return

    def save_alert(self, frame, *args, **kwargs):
        t = threading.Thread(target=self._save_alert, args=(frame,) + args, kwargs=kwargs)
        t.start()
        return t

    def update_video_alert(self, frame):
        self.video_recorder.process_frame(frame, time.time())


