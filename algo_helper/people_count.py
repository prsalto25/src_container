import cv2
cv2.setNumThreads(1)
from datetime import datetime, timedelta
import uuid
import os
import numpy as np

from .use import drawing
from .use.point_in_poly import point_in_poly
from ._alert_va import AlertsVA

class ThresholdDurationTrigger:
    def __init__(self, count_threshold=3, duration_threshold=5, cooldown_minutes=5):
        self.count_threshold = count_threshold
        self.duration_threshold = duration_threshold  # seconds
        self.cooldown_minutes = cooldown_minutes

        self.trigger_start_time = None
        self.last_alert_time = None

    def check(self, current_count):
        now = datetime.now()

        # If in cooldown, skip triggering
        if self.last_alert_time and (now - self.last_alert_time) < timedelta(minutes=self.cooldown_minutes):
            return False

        if current_count >= self.count_threshold:
            if self.trigger_start_time is None:
                self.trigger_start_time = now

            elapsed = (now - self.trigger_start_time).total_seconds()
            if elapsed >= self.duration_threshold:
                self.last_alert_time = now  # Set cooldown
                self.trigger_start_time = None  # Reset for next stable detection
                return True
        else:
            self.trigger_start_time = None  # Reset because people count dropped

        return False


class PeopleCount(AlertsVA):
    def __init__(self, algos_dict, outstream, mysql_helper):
        super().__init__(
            algos_dict,
            mysql_helper,
            sql_table_name="crowd_count",
            sql_table_fields=[
                ["track_id", "varchar(45)"],
                ["time", "datetime"],
                ["cam_id", "varchar(45)"],
                ["cam_name", "varchar(45)"],
                ["id_account", "varchar(45)"],
                ["id_branch", "varchar(45)"],
                ["people_count", "varchar(255)"],
            ],
            alert_folder_resource_path=f"/home/resources/{algos_dict['id_account']}/{algos_dict['id_branch']}/crowd_count/{algos_dict['camera_id']}/",
            video_recorder_buffer_seconds=3,
            video_recorder_record_seconds=3,
            video_alert_resolution=(640, 480),
            image_alert_resolution=(640, 480),
        )

        self.attr = algos_dict
        self.outstream = outstream
        self.mysql_helper = mysql_helper  # Even if not used now
        self.trigger_logic = ThresholdDurationTrigger(count_threshold=3, duration_threshold=5, cooldown_minutes=5)

    def _is_inside_roi(self, x, y):
        return point_in_poly(x, y, self.attr['rois'])

    # def _trigger_alert(self, frame):
    #     cv2.putText(frame, "ALERT: AREA HAS 3+ PEOPLE!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    def trigger_alert(self, frame, people_inside_roi):
        # print("trigger_alert people_counting")
        cv2.putText(frame, "ALERT: AREA HAS 3+ PEOPLE!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        people_count = str(people_inside_roi)
        date = datetime.now().strftime("%Y-%-m-%-d_%H:%M:%S")
        id_ = str(uuid.uuid4())
        self.save_alert(
            frame,
            image_fname=f"{date.replace(' ', '_', 1)}_{id_}.jpg",
            video_fname=f"{date}_{id_}_video.mp4",
            alert_data=(
                id_,
                date,
                self.attr["camera_id"],
                self.attr["camera_name"],
                self.attr["id_account"],
                self.attr["id_branch"],
                people_count
            ),
            tickets_data=(
                id_,
                "People Counting",
                date,
                date,
                "NULL",
                self.attr["id_account"],
                self.attr["id_branch"],
                1,
                self.attr["camera_name"],
                self.attr["camera_id"],
                os.path.join(
                    self.alert_folder_resource_path,
                    f"{date}_{id_}.jpg",
                ),
                "NULL",
                "NULL",
            ),
        )
        return

    def run(self, frame, yolo_person):
        if self.attr['rois'] is not None:
            drawing.draw_rois(frame, self.attr['rois'], (255, 0, 0))

        people_inside_roi = 0

        for track in yolo_person:
            (x1, y1, x2, y2), (x, y, w, h), tid = track.attr['xyxy'], track.attr['xywh'], str(track.id)

            if w > 1000:
                continue

            if self._is_inside_roi(x, y):
                people_inside_roi += 1
                drawing.plot_one_box(frame, (x1, y1), (x2, y2), color=(255, 0, 0), label=f"ID: {tid}", line_thickness=2)
            else:
                drawing.plot_one_box(frame, (x1, y1), (x2, y2), color=(125, 125, 125), label=f"ID: {tid}", line_thickness=2)

        cv2.putText(frame, f"People in ROI: {people_inside_roi}", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        if self.trigger_logic.check(people_inside_roi):
            self.trigger_alert(frame, people_inside_roi)
            print("trigger")

        self.outstream.write(frame)
