import os
import re
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
import time
import cv2
import numpy as np
from typing import Tuple, Literal

from .use import drawing
from .use.point_in_poly import point_in_poly
from ._alert_va import AlertsVA
from .use.various_dicts import LimitedSizeDict

class VehicleEnterExit(AlertsVA):
    def __init__(self, algos_dict, outstream, mysql_helper):
        super().__init__(
            algos_dict,
            mysql_helper,
            sql_table_name="vcount",
            sql_table_fields=[
                ["track_id", "varchar(45)"],
                ["time", "datetime"],
                ["cam_id", "varchar(45)"],
                ["cam_name", "varchar(45)"],
                ["id_account", "varchar(45)"],
                ["id_branch", "varchar(45)"],
                ["vehicle_type", "varchar(45)"],
                ["alert", "varchar(45)"],
                ["car_numbers", "int"],
                ["motorbike_numbers", "int"],
                ["truck_numbers", "int"],
                ["bus_numbers", "int"],
            ],
            alert_folder_resource_path=f"/home/resources/{algos_dict['id_account']}/{algos_dict['id_branch']}/vcount/{algos_dict['camera_id']}/",
            video_recorder_buffer_seconds=3,
            video_recorder_record_seconds=3,
            video_alert_resolution=(640, 480),
            image_alert_resolution=(640, 480),
        )
        self.outstream = outstream
        # self.track_memory = LimitedSizeDict(max_size=20)
        self.track_position_state = {}
        self.enter_counter = 0
        self.exit_counter = 0
        self.test_alert = time.time()

    def trigger_alert(self, frame, vehicle_type, alert_string):
        print("trigger_alert vehicle_enter_exit")
        # vehicle_type = "VEHICLE_TYPE_TEST"
        # alert_string = "ALERT_STRING_TEST"
        car_numbers = 0
        motorbike_numbers = 0
        truck_numbers = 0
        bus_numbers = 0
        date = datetime.now().strftime("%Y-%-m-%-d_%H:%M:%S")
        id_ = str(uuid.uuid4())
        self.save_alert(
            frame,
            image_fname=f"{date.replace(' ', '_', 1)}_undefined.jpg",
            video_fname=f"{date}_undefined_video.mp4",
            alert_data=(
                id_,
                date,
                self.attr["camera_id"],
                self.attr["camera_name"],
                self.attr["id_account"],
                self.attr["id_branch"],
                vehicle_type,
                alert_string,
                car_numbers,
                motorbike_numbers,
                truck_numbers,
                bus_numbers
            ),
            tickets_data=(
                id_,
                "Vehicle Counting",
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
                    f"{date}_undefined.jpg",
                ),
                "NULL",
                "NULL",
            ),
        )
        return

    def is_bbox_inside_rois(
        self,
        frame: np.ndarray,
        xyxy: Tuple[int, int, int, int],
        mode: Literal["intersect", "inside", "centroid"],
    ) -> bool:
        rois = self.attr.get("rois")
        if not rois:
            return True
        x1, y1, x2, y2 = xyxy
        bbox_poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        for roi in rois:
            roi_np = np.array(roi, dtype=np.int32)
            if mode == "centroid":
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if cv2.pointPolygonTest(roi_np, (cx, cy), False) >= 0:
                    return True
            elif mode == "inside":
                inside = all(
                    cv2.pointPolygonTest(roi_np, tuple(point), False) >= 0
                    for point in bbox_poly
                )
                if inside:
                    return True
            elif mode == "intersect":
                mask_shape = frame.shape[:2]
                roi_mask = np.zeros(mask_shape, dtype=np.uint8)
                bbox_mask = np.zeros(mask_shape, dtype=np.uint8)

                cv2.fillPoly(roi_mask, [roi_np], 255)
                cv2.fillPoly(bbox_mask, [bbox_poly], 255)

                intersection = cv2.bitwise_and(roi_mask, bbox_mask)
                if np.any(intersection):
                    return True
        return False

    def _is_inside_roi(self, x, y):
        return point_in_poly(x, y, self.attr['rois'])

    # START EVERYTHING HERE !!!

    def run(self, frame, vehicle_yolo):

        # ALERT TEST !!!
        if time.time() - self.test_alert > 60:  # send alert every 2 mins
            # self.trigger_alert(frame)
            self.test_alert = time.time()
        # ALERT TEST END

        if self.attr['rois'] is not None:
            drawing.draw_rois(frame, self.attr['rois'], (255, 0, 0))

        people_inside_roi = 0

        for track in vehicle_yolo:
            # print(track.attr)
            (x1, y1, x2, y2), (x, y, w, h), tid, vehicle_type = track.attr['xyxy'], track.attr['xywh'], str(track.id), track.attr['cls']

            if w > 1000:
                continue

            is_inside = self._is_inside_roi(x, y)
            current_state = "inside" if is_inside else "outside"
            prev_state = self.track_position_state.get(tid)

            if prev_state == "outside" and current_state == "inside":
                print("enter")
                self.enter_counter += 1
                # self.trigger_alert(vehicle_type, "Enter")
            elif prev_state == "inside" and current_state == "outside":
                print("exit")
                self.exit_counter += 1
                # self.trigger_alert(vehicle_type, "Exit")

            self.track_position_state[tid] = current_state
            # current_tids.add(tid)

            if self._is_inside_roi(x, y):
                people_inside_roi += 1
                if vehicle_type == 'car':
                    drawing.plot_one_box(frame, (x1, y1), (x2, y2), color=(255, 0, 0), label=f"CAR ID: {tid}", line_thickness=2)
                elif vehicle_type == 'motorcycle':
                    drawing.plot_one_box(frame, (x1, y1), (x2, y2), color=(0, 200, 100), label=f"MTR ID: {tid}", line_thickness=2)
                elif vehicle_type == 'bus':
                    drawing.plot_one_box(frame, (x1, y1), (x2, y2), color=(255, 0, 200), label=f"BUS ID: {tid}", line_thickness=2)
                elif vehicle_type == 'truck':
                    drawing.plot_one_box(frame, (x1, y1), (x2, y2), color=(255, 200, 0), label=f"TRC ID: {tid}", line_thickness=2)
                else:
                    drawing.plot_one_box(frame, (x1, y1), (x2, y2), color=(0, 0, 255), label=f"UNK ID: {tid}", line_thickness=2)
            else:
                drawing.plot_one_box(frame, (x1, y1), (x2, y2), color=(125, 125, 125), label=f"ID: {tid}", line_thickness=2)

        cv2.putText(frame, f"Vehicle in ROI: {people_inside_roi}", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, f"Enter counter: {self.enter_counter}", (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, f"Exit counter: {self.exit_counter}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        # if self.trigger_logic.check(people_inside_roi):
        #     self.trigger_alert(frame, people_inside_roi)
        #     print("trigger")

        # self.outstream.write(frame)

        self.outstream.write(frame)
        self.update_video_alert(frame)
        return frame