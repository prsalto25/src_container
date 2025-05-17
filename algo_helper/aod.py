import cv2
cv2.setNumThreads(1)
from datetime import datetime, timedelta
from math import hypot
import uuid
import os
import numpy as np

from ._alert_va import AlertsVA


class AODTriggerLogic:
    def __init__(self, left_duration=10, cooldown_minutes=5):
        self.left_duration = left_duration  # Seconds before triggering after people leave
        self.cooldown_minutes = cooldown_minutes
        self.records = []  # [{pos, last_seen_time, left_time, cooldown_until}]

    def _is_same_bag(self, pos1, pos2, threshold=30):
        return hypot(pos1[0] - pos2[0], pos1[1] - pos2[1]) < threshold

    def _is_near_person(self, bag_center, people, threshold=50):
        for person in people:
            (x1, y1, x2, y2), (px, py, w, h) = person.attr['xyxy'], person.attr['xywh']
            if hypot(bag_center[0] - px, bag_center[1] - py) < threshold:
                return True
        return False

    def _find_record(self, bag_center):
        for record in self.records:
            if self._is_same_bag(bag_center, record['pos']):
                return record
        return None

    def _should_alert(self, record):
        now = datetime.now()
        if record['cooldown_until'] and now < record['cooldown_until']:
            return False

        if record['left_time']:
            elapsed = (now - record['left_time']).total_seconds()
            if elapsed >= self.left_duration:
                record['cooldown_until'] = now + timedelta(minutes=self.cooldown_minutes)
                return True

        return False

    def update_and_check(self, bag_center, people):
        now = datetime.now()
        record = self._find_record(bag_center)

        if record:
            # Check if any person is still near
            if self._is_near_person(bag_center, people):
                record['left_time'] = None  # Reset
            else:
                if record['left_time'] is None:
                    record['left_time'] = now  # Start the countdown

            return self._should_alert(record)

        else:
            # New bag found
            self.records.append({
                'pos': bag_center,
                'left_time': None,
                'cooldown_until': None,
            })
            return False


class AOD(AlertsVA):
    def __init__(self, algos_dict, outstream, mysql_helper):
        super().__init__(
            algos_dict,
            mysql_helper,
            sql_table_name="aod",
            sql_table_fields=[
                ["time", "datetime"],
                ["zone", "int"],
                ["cam_name", "varchar(45)"],
                ["id", "varchar(45)"],
                ["cam_id", "varchar(45)"],
                ["id_account", "varchar(45)"],
                ["id_branch", "varchar(45)"],
                ["track_id", "varchar(45)"],
            ],
            alert_folder_resource_path=f"/home/resources/{algos_dict['id_account']}/{algos_dict['id_branch']}/aod/{algos_dict['camera_id']}/",
            video_recorder_buffer_seconds=3,
            video_recorder_record_seconds=3,
            video_alert_resolution=(640, 480),
            image_alert_resolution=(640, 480),
        )

        left_duration = 10
        cooldown_minutes = 5
        self.outstream = outstream
        self.trigger_logic = AODTriggerLogic(left_duration, cooldown_minutes)

    def trigger_alert(self, frame):
        print("trigger_alert abandoned_object")
        zone = 0
        date = datetime.now().strftime("%Y-%-m-%-d_%H:%M:%S")
        id_ = str(uuid.uuid4())
        self.save_alert(
            frame,
            image_fname=f"{date.replace(' ', '_', 1)}_{id_}.jpg",
            video_fname=f"{date}_{id_}_video.mp4",
            alert_data=(
                date,
                zone,
                self.attr["camera_name"],
                id_,
                self.attr["camera_id"],
                self.attr["id_account"],
                self.attr["id_branch"],
                id_,
            ),
            tickets_data=(
                id_,
                "Abandoned Object",
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

    def run(self, frame, yolo_person, yolo_bag):
        for bag in yolo_bag:
            x1, y1, x2, y2 = map(int, bag['xyxy'])
            x, y, w, h = map(int, bag['xywh'])
            bag_center = (x, y)

            # Draw the bag
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 0), 2)

            if self.trigger_logic.update_and_check(bag_center, yolo_person):
                # print('trigger aod')
                self.trigger_alert(frame)
                cv2.putText(frame, "ALERT: ABANDONED OBJECT!", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.circle(frame, bag_center, 20, (0, 0, 255), 3)

        self.outstream.write(frame)
