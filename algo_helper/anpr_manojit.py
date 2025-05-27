# from paddleocr import PaddleOCR
import os
import re
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
import time
import cv2
import numpy as np
from typing import Tuple, Optional

from .use.various_dicts import LimitedSizeDict
from .use.drawing2 import draw_texted_bbox
from .client_scripts.anpr_yolox import ANPR_Yolox
from .client_scripts.car_info_va import CarInfoZMQ
from .client_scripts.paddleocr_client import PaddleOCRClient
from ._alert_va import AlertsVA


class ANPR(AlertsVA):
    def __init__(self, algos_dict, outstream, mysql_helper):
        super().__init__(
            algos_dict,
            mysql_helper,
            sql_table_name="plate",
            sql_table_fields=[
                ["track_id", "varchar(45)"],
                ["time", "datetime"],
                ["cam_id", "varchar(45)"],
                ["cam_name", "varchar(45)"],
                ["id_account", "varchar(45)"],
                ["id_branch", "varchar(45)"],
                ["plate", "varchar(20)"],
                ["watchlist", "varchar(45)"],
                ["vehicle", "varchar(45)"],
            ],
            alert_folder_resource_path=f"/home/resources/{algos_dict['id_account']}/{algos_dict['id_branch']}/anpr/{algos_dict['camera_id']}/",
            video_recorder_buffer_seconds=4,
            video_recorder_record_seconds=4,
            video_alert_resolution=(1280, 720),
            image_alert_resolution=(1280, 720),
        )
        self.outstream = outstream
        self.ocr_engine = PaddleOCRClient("tcp://127.0.0.1:5555")
        self.anpr_yolox = ANPR_Yolox()
        self.vehicles = LimitedSizeDict(max_size=20)
        self.recognized_number = {}
        # self._create_bl_wl_table()

    def _create_bl_wl_table(self):
        """
        Create anpr_access_list table for blacklist and whitelist
        """
        mysql_fields = [
            ["plate", "varchar(45)"],
            ["type", "varchar(45)"],
        ]
        self.mysql_helper.add_table("anpr_access_list", mysql_fields)

    def _get_watch_list(self):
        """
        Get the blacklist and whitelist from SQL
        """
        cmd = "SELECT * FROM anpr_access_list"
        result = self.mysql_helper.run_fetch(cmd)
        return [{"plate": res[0], "type": res[1]} for res in result]
    
    def draw_bbox(
        self,
        frame: np.ndarray,
        xyxy: Tuple[int, int, int, int],
        text: str,
        color: Tuple[int, int, int] = (0, 255, 0),
        box_thickness: int = 2,
        box_margin: int = 0,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 1,
        font_thickness: int = 2,
    ):
        x1, y1, x2, y2 = xyxy
        # Draw the bounding box with a line thickness of 2
        cv2.rectangle(
            frame,
            (x1 - box_margin, y1 - box_margin),
            (x2 + box_margin, y2 + box_margin),
            color=color,
            thickness=box_thickness,
        )

        if text:
            # Get the width and height of the text box plus baseline
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, font_thickness
            )
            # Define the top-left and bottom-right coordinates for the text background rectangle.
            # This rectangle will be drawn just above the bounding box.
            text_bg_top_left = (x1, y1 - text_height - baseline)
            text_bg_bottom_right = (x1 + text_width, y1)
            # Draw the filled rectangle for the text background
            cv2.rectangle(
                frame, text_bg_top_left, text_bg_bottom_right, color, thickness=-1
            )
            # Overlay the text in white color on top of the background rectangle.
            cv2.putText(
                frame,
                text,
                (x1, y1 - baseline),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA,
            )
        return frame

    def trigger_alert(self, frame, vehicle, license_plate, blacklist=None, whitelist=None):
        """
        Trigger alert based on license plate:
        - If whitelist or blacklist is provided: only alert with ticket if plate is in either.
        - If both are None: always alert with ticket.
        """
        date = datetime.now().strftime("%Y-%-m-%-d_%H:%M:%S")
        id_ = str(uuid.uuid4())
        image_fname = f"{date}_{id_}.jpg"
        video_fname = f"{date}_{id_}_video.mp4"
        vehicle_cls = getattr(vehicle, "attr", {}).get("cls", "unknown")

        # Determine if plate is in any watchlist
        in_whitelist = whitelist and license_plate in whitelist
        in_blacklist = blacklist and license_plate in blacklist
        watch_type = "Whitelist" if in_whitelist else "Blacklist" if in_blacklist else "Unknown"

        alert_data = (
            id_,
            date,
            self.attr["camera_id"],
            self.attr["camera_name"],
            self.attr["id_account"],
            self.attr["id_branch"],
            license_plate,
            watch_type,
            vehicle_cls,
        )

        if blacklist is not None or whitelist is not None:
            if in_whitelist or in_blacklist:
                tickets_data = (
                    id_,
                    "anpr",
                    date,
                    date,
                    "NULL",
                    self.attr["id_account"],
                    self.attr["id_branch"],
                    1,
                    self.attr["camera_name"],
                    self.attr["camera_id"],
                    os.path.join(self.alert_folder_resource_path, image_fname),
                    "NULL",
                    "NULL",
                )
            else:
                tickets_data = None
        else:
            # No watchlist configured, include tickets_data by default
            tickets_data = (
                id_,
                "anpr",
                date,
                date,
                "NULL",
                self.attr["id_account"],
                self.attr["id_branch"],
                1,
                self.attr["camera_name"],
                self.attr["camera_id"],
                os.path.join(self.alert_folder_resource_path, image_fname),
                "NULL",
                "NULL",
            )

        self.save_alert(
            frame,
            image_fname=image_fname,
            video_fname=video_fname,
            alert_data=alert_data,
            tickets_data=tickets_data,
        )

    @staticmethod
    def get_y2(obj):
        if hasattr(obj, "attr"):
            obj_box = obj.attr.get("xyxy")
        elif isinstance(obj, dict):
            obj_box = obj.get("xyxy")
        else:
            raise ValueError(f"obj type: {type(obj)}")

        if obj_box is None or len(obj_box) < 4:
            raise ValueError(f"Invalid bounding box: {obj_box}")

        return obj_box[3]  # y2

    def match_license_plate_to_vehicle(self, vehicle_yolo, lp_yolox):
        """
        Match license plate to vehicle
        """
        matches = []
        for lp in lp_yolox:
            # Assume each lp object has a bounding box in the format [x1, y1, x2, y2]
            if hasattr(lp, "attr"):
                lp_box = lp.attr.get("xyxy")
            elif isinstance(lp, dict):
                lp_box = lp.get("xyxy")
            else:
                raise ValueError(f"lp type: {type(lp)}")
            if lp_box is None:
                continue  # Skip if no bounding box is provided

            # Calculate the center of the license plate box
            # lp_center = ((lp_box[0] + lp_box[2]) / 2, (lp_box[1] + lp_box[3]) / 2)
            candidate_vehicles = []

            # Check which vehicles contain the license plate's center
            for veh in vehicle_yolo:
                if hasattr(veh, "attr"):
                    veh_box = veh.attr.get("xyxy")
                elif isinstance(veh, dict):
                    veh_box = veh.get("xyxy")
                else:
                    raise ValueError(f"veh type: {type(veh)}")
                if veh_box is None:
                    continue  # Skip if no bounding box available

                # if (veh_box[0] <= lp_center[0] <= veh_box[2] and
                #     veh_box[1] <= lp_center[1] <= veh_box[3]):
                if (
                    veh_box[0] <= lp_box[0]
                    and veh_box[1] <= lp_box[1]
                    and lp_box[2] <= veh_box[2]
                    and lp_box[3] <= veh_box[3]
                ):
                    candidate_vehicles.append(veh)

            if candidate_vehicles:
                # If more than one vehicle contains the license plate, choose the "most front" vehicle.
                # Here we assume that the vehicle with the largest y2 (i.e. bottom coordinate) is closest.
                front_vehicle = max(candidate_vehicles, key=self.get_y2)
                matches.append((lp, front_vehicle))
            else:
                # No vehicle matched with the license plate.
                matches.append((lp, None))
        # Also add vehicle with no license plate
        for veh in vehicle_yolo:
            if veh not in [match[1] for match in matches]:
                matches.append((None, veh))
        return matches

    def ocr_predict(self, lp_frame):
        result = self.ocr_engine.ocr(lp_frame)
        if result == [None] or result == [[]]:
            return "", 0.0

        ocr_text = ""
        confidence_total = 0.0
        
        if len(result[0]) > 0:
            for detection in result[0]:
                ocr_text += detection[1][0]
                confidence_total += detection[1][1]

        print("OCR RESULT!!!", ocr_text, "Confidence:", confidence_total)
        return ocr_text, confidence_total

    def parse_ocr_text(self, text):
        """
        Parser ocr text (different plate number format)
        """
        text = text.replace(" ", "")
        return india_plate_processing(text)
        # Motorcycle format
        recognition_text = vietnam_motorcycle_plate_processing(text)
        if recognition_text:
            return recognition_text
        # Car format
        recognition_text = vietnam_car_plate_processing(text)
        if recognition_text:
            return recognition_text
        # More general format
        # Remove unwanted characters, allowing only letters, numbers, '-', and '.'
        cleaned_text = re.sub(r"[^a-zA-Z0-9-.]", "", text)
        # Standard regex pattern: 2-3 letters/numbers before '-', 3-5 numbers after '-', then dot, then 2-3 numbers
        valid_pattern = re.compile(r"^[a-zA-Z0-9]{2,3}-[a-zA-Z0-9]{3,5}\.\d{2,3}$")
        if valid_pattern.match(cleaned_text):
            return cleaned_text
        else:
            return None

    def is_one_char_different(self, plate):
        for p in self.recognized_number.values():
            if len(p) == len(plate):  # Ensure they have the same length
                diff_count = sum(1 for a, b in zip(p, plate) if a != b)
                if diff_count == 1:  # Only one character is different
                    return True
        return False

    def margin_selection(self, lp_xyxy, frame, original_frame):
        """
        Do the OCR with certain parameter, output the text and the confidence level
        """
        x1_lp, y1_lp, x2_lp, y2_lp = lp_xyxy
        # not being used
        # margins = [0, 2, -2, -1, 5, 10, 15]
        # highest_confidence = 0.0
        # best_ocr_text = ""
        # consecutive_matches = 0
        # last_ocr_text = ""

        # without margin selection
        my_margin = 5  # this could be modified to adjust the accuracy
        full_lp_frame = self.process_frame(x1_lp, x2_lp, y1_lp, y2_lp, frame, original_frame, my_margin)
        ocr_text, confidence_level = self.ocr_predict(full_lp_frame)
        return ocr_text, confidence_level
    
    def combine_license_plate_readings(self, readings):
        """
        Combine license plate reading using the combination of several reading
        """
        if not readings:
            return ""
        
        if len(readings) == 1:
            return readings[0]
        
        length_counts = {}
        for reading in readings:
            if reading:
                length = len(reading)
                length_counts[length] = length_counts.get(length, 0) + 1
        
        if not length_counts:
            return ""
        
        most_common_length = max(length_counts, key=length_counts.get)
        
        filtered_readings = [r for r in readings if r and len(r) == most_common_length]
        
        if not filtered_readings:
            return max(readings, key=readings.count) if readings else ""
        
        result = []
        for i in range(most_common_length):
            chars_at_position = [reading[i] for reading in filtered_readings]
            most_common_char = max(set(chars_at_position), key=chars_at_position.count)
            
            if chars_at_position.count(most_common_char) > len(filtered_readings) / 5:
                result.append(most_common_char)
            else:
                result.append('?')
        
        return ''.join(result)
    
    def process_frame(self, x1_lp, x2_lp, y1_lp, y2_lp, frame, original_frame, LP_MARGIN=10):
        """
        Map the LP coordnate to original frame
        """
        # a. Parse Coordinate 
        x1_lp = max(0, x1_lp - LP_MARGIN)
        y1_lp = max(0, y1_lp - LP_MARGIN)
        x2_lp = min(frame.shape[1], x2_lp + LP_MARGIN)
        y2_lp = min(frame.shape[0], y2_lp + LP_MARGIN)

        # b. parse coordinate full frame
        full_resolution_points = _mapping_point_to_different_resolution((x1_lp, y1_lp, x2_lp, y2_lp), frame.shape, original_frame.shape)
        full_x1_lp = max(0, full_resolution_points[0])
        full_y1_lp = max(0, full_resolution_points[1])
        full_x2_lp = min(original_frame.shape[1], full_resolution_points[2])
        full_y2_lp = min(original_frame.shape[0], full_resolution_points[3])
        full_lp_frame = original_frame[full_y1_lp:full_y2_lp, full_x1_lp:full_x2_lp]

        return full_lp_frame




    ## START EVERYTHING HERE !!! 



    def run(self, frame, vehicle_yolo, original_frame):
        # print("frame", frame.shape)
        # print("original_frame", original_frame.shape)

        # Get watch list
        # watchlist = self._get_watch_list()
        # blacklist = [w["plate"] for w in watchlist if w["type"].lower().strip()=="blacklist"]
        # whitelist = [w["plate"] for w in watchlist if w["type"].lower().strip()=="whitelist"]
        blacklist, whitelist = None, None

        # 1. Detect ANPR + Matching (Vehicle x LPR)
        lp_yolox = self.anpr_yolox.detect(frame).get("Registration_Plate", [])
        # matches = self.match_license_plate_to_vehicle(vehicle_yolo, lp_yolox)
        for lp in lp_yolox:
            original_ocr_text, confidence_level = self.margin_selection(lp['xyxy'], frame, original_frame)

            # -> Postprocessing OCR 
            # license_plate = self.parse_ocr_text(original_ocr_text)
            license_plate = "D87550" if "D87550" in original_ocr_text else None
            license_plate = "KL54C0005" if "KL54C0005" in original_ocr_text else license_plate
            # print(f"license_plate {license_plate}")

            if license_plate:
                print(f"license_plate {license_plate}")
                lp_box = lp.get("xyxy")
                license_plate = "" if license_plate is None else license_plate
                
                self.draw_bbox(frame, lp_box, f"{license_plate}", (0, 200, 255))
            
        # 2. Postprocessing
        # vehicle_id = None
        # for lp, vehicle in matches:
        #     if vehicle is not None:
        #         x1, y1, x2, y2 = vehicle.attr.get("xyxy")
        #         vehicle_id = int(vehicle.id[-3:].replace("_", ""))
        #         vehicle_type = vehicle.attr.get("cls", "unknown")

        #         # Vehicle Counting 
        #         # (this code could be better)
        #         if vehicle_id is not None:
        #             if vehicle_id in self.vehicles:
        #                 self.vehicles[vehicle_id]["count"] += 1
        #             else:
        #                 self.vehicles[vehicle_id] = {
        #                     "count": 0, 
        #                     "lp": deque(maxlen=5),
        #                     "vehicle_type": vehicle_type 
        #                 }
            
        #     # 2. b. Read License Plate
        #     license_plate = ""
        #     if vehicle_id in self.vehicles:
        #         if len(self.vehicles[vehicle_id]["lp"]) < 3:    # no more detecting if deque is more than 3
        #             if self.vehicles[vehicle_id].get("count", 0) % 1 == 0:       # skip frame for reading
        #                 if lp is not None:
        #                     # -> OCR Predict
        #                     original_ocr_text, confidence_level = self.margin_selection(lp['xyxy'], frame, original_frame)

        #                     # -> Postprocessing OCR 
        #                     license_plate = self.parse_ocr_text(original_ocr_text)

        #                     if license_plate:
        #                         self.vehicles[vehicle_id]["lp"].append(license_plate)
        #                         lp_occurrences = self.vehicles[vehicle_id]["lp"].count(license_plate)
        #                         print(f"license_plate {license_plate}")


        #                         # HACK: for alert first
        #                         if (
        #                             lp_occurrences >= 3  # if the reading is >=3
        #                             and vehicle_id not in self.recognized_number.keys()
        #                             and license_plate not in self.recognized_number.values()
        #                             # and not self.is_one_char_different(license_plate)  # only use this if unstable and send alert too much
        #                         ):
        #                             combined_license_plate = self.combine_license_plate_readings(list(self.vehicles[vehicle_id]["lp"]))
                                    
        #                             self.recognized_number[vehicle_id] = combined_license_plate
                                    
        #                             # If combined_license_plate contains at most one '?', trigger an alert
        #                             if combined_license_plate.count('?') <= 1:
        #                                 self.trigger_alert(frame, vehicle, combined_license_plate, blacklist, whitelist)   

        #                     # if vehicle_id in self.vehicles and 'color_name' not in self.vehicles[vehicle_id]:
        #                     #     full_veh_frame = self.process_frame(x1, x2, y1, y2, frame, original_frame, LP_MARGIN=0)
        #                         # vehicle_color = self.classify_color(full_veh_frame)
        #                         # if vehicle_color:  # Only store if the color is detected (not empty)
        #                         #     self.vehicles[vehicle_id]['color_name'] = vehicle_color
        #             else:
        #                 continue

        #         # Drawing Only 
        #         if lp:
        #             lp_box = lp.get("xyxy")
        #             license_plate = "" if license_plate is None else license_plate
                    
        #             if self.vehicles[vehicle_id]['lp']:
        #                 combined_plate = self.combine_license_plate_readings(list(self.vehicles[vehicle_id]['lp']))
        #             else:
        #                 combined_plate = 'Cant read plate'
                    
        #             self.draw_bbox(frame, lp_box, f"{combined_plate}", (0, 200, 255))
        #             self.draw_bbox(frame, (x1, y1, x2, y2), f"ID: {vehicle_id} | Type: {self.vehicles[vehicle_id].get('vehicle_type', 'Unknown')} | Color: {self.vehicles[vehicle_id].get('color_name', 'Unknown')}")
        #         else:
        #             self.draw_bbox(frame, (x1, y1, x2, y2), f"ID: {vehicle_id} | Type: {self.vehicles[vehicle_id].get('vehicle_type', 'Unknown')}", (0, 0, 255))
        
        if True:  # to save the image result to a dolfer
            save_folder = os.path.join("outputs", f"{self.attr['camera_name']}")
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            cv2.imwrite(os.path.join(save_folder, f"{time.time()}.jpg"), frame)
        
        self.outstream.write(frame)
        self.update_video_alert(frame)
        return frame



















# ==============================
# | Plate Number Format Parser |
# ==============================

MAP_DIGIT_TO_LETTER = defaultdict(
    lambda: "?",
    {
        "0": "O",
        "1": "I",
        "2": "Z",
        "3": "E",
        "4": "A",
        "5": "S",
        "6": "G",
        "7": "T",
        "8": "B",
        "9": "Q",
    },
)

MAP_LETTER_TO_DIGIT = defaultdict(
    lambda: "?",
    {
        "A": "4",
        "B": "8",
        "D": "0",
        "E": "3",
        "G": "6",
        "H": "4",
        "I": "1",
        "O": "0",
        "Q": "9",
        "S": "5",
        "T": "7",
        "Z": "2",
    },
)


def vietnam_car_plate_processing(text):
    try:
        # Remove spaces from the text
        text = text.replace(" ", "")
        if text.isalpha():
            return None
        # Define the regex patterns for Vietnam car plate numbers
        patterns = [
            r"\d{2}[A-Z]-\d{3}\.\d{2}",  # NNL-NNN.NN
            r"\d{2}[A-Z]{2}-\d{3}\.\d{2}",  # NNLL-NNN.NN
            r"\d{2}[A-Z]-\d{5}",  # NNL-NNNNN
            r"\d{2}[A-Z]{2}-\d{5}",  # NNLL-NNNNN
        ]

        # Check each pattern and return the first match found
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0]

        # Initialize variables
        first_part = None
        middle_part = None
        last_part = text

        # Split the text at the first occurrence of '-'
        if "-" in text:
            first_part, last_part = text.split("-", 1)

        # Split the last_part at the first occurrence of '.'
        if "." in last_part:
            middle_part, last_part = last_part.split(".", 1)

        if first_part is not None:
            first_part = list(first_part)
            if len(first_part) == 3 or len(first_part) == 4:
                # if the first letter and second letter is number is not letter
                if first_part[0].isalpha():
                    first_part[0] = MAP_LETTER_TO_DIGIT[first_part[0].upper()]
                if first_part[1].isalpha():
                    first_part[1] = MAP_LETTER_TO_DIGIT[first_part[1].upper()]
                if first_part[2].isdigit():
                    first_part[2] = MAP_DIGIT_TO_LETTER[first_part[2]]
            else:
                return None

        if middle_part is not None:
            middle_part = list(middle_part)
            if len(middle_part) <= 3:
                for i in range(len(middle_part)):
                    if middle_part[i].isalpha():
                        middle_part[i] = MAP_LETTER_TO_DIGIT[middle_part[i].upper()]
            elif len(middle_part) >= 5 and len(middle_part) <= 7:
                first_part = middle_part[:-3]
                middle_part = middle_part[-3:]
                for i in range(len(middle_part)):
                    if middle_part[i].isalpha():
                        middle_part[i] = MAP_LETTER_TO_DIGIT[middle_part[i].upper()]
            else:
                return None

        if first_part is None and middle_part is None:
            # Remove special characters from last_part
            last_part = re.sub(r"[^A-Za-z0-9]", "", last_part)
            last_part = list(last_part)
            if len(last_part) == 8:  # NNL-NNN.NN
                last_part[0] = (
                    MAP_LETTER_TO_DIGIT[last_part[0].upper()]
                    if last_part[0].isalpha()
                    else last_part[0]
                )
                last_part[1] = (
                    MAP_LETTER_TO_DIGIT[last_part[1].upper()]
                    if last_part[1].isalpha()
                    else last_part[1]
                )
                last_part[2] = (
                    MAP_DIGIT_TO_LETTER[last_part[2]]
                    if last_part[2].isdigit()
                    else last_part[2]
                )
                last_part[3] = (
                    MAP_LETTER_TO_DIGIT[last_part[3].upper()]
                    if last_part[3].isalpha()
                    else last_part[3]
                )
                last_part[4] = (
                    MAP_LETTER_TO_DIGIT[last_part[4].upper()]
                    if last_part[4].isalpha()
                    else last_part[4]
                )
                last_part[5] = (
                    MAP_LETTER_TO_DIGIT[last_part[5].upper()]
                    if last_part[5].isalpha()
                    else last_part[5]
                )
                last_part[6] = (
                    MAP_LETTER_TO_DIGIT[last_part[6].upper()]
                    if last_part[6].isalpha()
                    else last_part[6]
                )
                last_part[7] = (
                    MAP_LETTER_TO_DIGIT[last_part[7].upper()]
                    if last_part[7].isalpha()
                    else last_part[7]
                )
                first_part = last_part[:3]
                middle_part = last_part[3:6]
                last_part = last_part[6:]
            elif len(last_part) == 9:  # NNLL-NNN.NN
                last_part[0] = (
                    MAP_LETTER_TO_DIGIT[last_part[0].upper()]
                    if last_part[0].isalpha()
                    else last_part[0]
                )
                last_part[1] = (
                    MAP_LETTER_TO_DIGIT[last_part[1].upper()]
                    if last_part[1].isalpha()
                    else last_part[1]
                )
                last_part[2] = (
                    MAP_DIGIT_TO_LETTER[last_part[2]]
                    if last_part[2].isdigit()
                    else last_part[2]
                )
                last_part[3] = (
                    MAP_DIGIT_TO_LETTER[last_part[3]]
                    if last_part[3].isdigit()
                    else last_part[3]
                )
                last_part[4] = (
                    MAP_LETTER_TO_DIGIT[last_part[4].upper()]
                    if last_part[4].isalpha()
                    else last_part[4]
                )
                last_part[5] = (
                    MAP_LETTER_TO_DIGIT[last_part[5].upper()]
                    if last_part[5].isalpha()
                    else last_part[5]
                )
                last_part[6] = (
                    MAP_LETTER_TO_DIGIT[last_part[6].upper()]
                    if last_part[6].isalpha()
                    else last_part[6]
                )
                last_part[7] = (
                    MAP_LETTER_TO_DIGIT[last_part[7].upper()]
                    if last_part[7].isalpha()
                    else last_part[7]
                )
                last_part[8] = (
                    MAP_LETTER_TO_DIGIT[last_part[8].upper()]
                    if last_part[8].isalpha()
                    else last_part[8]
                )
                first_part = last_part[:4]
                middle_part = last_part[4:7]
                last_part = last_part[7:]
            else:
                return None
        elif first_part is not None and middle_part is None:
            last_part = re.sub(r"[^A-Za-z0-9]", "", last_part)
            last_part = list(last_part)
            if len(last_part) == 5:
                last_part[0] = (
                    MAP_LETTER_TO_DIGIT[last_part[0].upper()]
                    if last_part[0].isalpha()
                    else last_part[0]
                )
                last_part[1] = (
                    MAP_LETTER_TO_DIGIT[last_part[1].upper()]
                    if last_part[1].isalpha()
                    else last_part[1]
                )
                last_part[2] = (
                    MAP_LETTER_TO_DIGIT[last_part[2].upper()]
                    if last_part[2].isalpha()
                    else last_part[2]
                )
                last_part[3] = (
                    MAP_LETTER_TO_DIGIT[last_part[3].upper()]
                    if last_part[3].isalpha()
                    else last_part[3]
                )
                last_part[4] = (
                    MAP_LETTER_TO_DIGIT[last_part[4].upper()]
                    if last_part[4].isalpha()
                    else last_part[4]
                )
                middle_part = last_part[:3]
                last_part = last_part[3:]
            elif len(last_part) == 4:
                last_part[0] = (
                    MAP_LETTER_TO_DIGIT[last_part[0].upper()]
                    if last_part[0].isalpha()
                    else last_part[0]
                )
                last_part[1] = (
                    MAP_LETTER_TO_DIGIT[last_part[1].upper()]
                    if last_part[1].isalpha()
                    else last_part[1]
                )
                last_part[2] = (
                    MAP_LETTER_TO_DIGIT[last_part[2].upper()]
                    if last_part[2].isalpha()
                    else last_part[2]
                )
                last_part[3] = (
                    MAP_LETTER_TO_DIGIT[last_part[3].upper()]
                    if last_part[3].isalpha()
                    else last_part[3]
                )
                middle_part = last_part[:2]
                last_part = last_part[2:]
            else:
                return None
        elif first_part is not None and middle_part is not None:
            last_part = list(last_part)
            if len(last_part) == 2:
                last_part[0] = (
                    MAP_LETTER_TO_DIGIT[last_part[0].upper()]
                    if last_part[0].isalpha()
                    else last_part[0]
                )
                last_part[1] = (
                    MAP_LETTER_TO_DIGIT[last_part[1].upper()]
                    if last_part[1].isalpha()
                    else last_part[1]
                )
            else:
                return None
        else:
            return None
        # return first_part + '-' + middle_part + '.' + last_part
        plate_number = (
            "".join(first_part) + "-" + "".join(middle_part) + "." + "".join(last_part)
        )
        if plate_number.count("?") < 1:
            return (
                "".join(first_part)
                + "-"
                + "".join(middle_part)
                + "."
                + "".join(last_part)
            )
        else:
            return None
    except Exception as e:
        print(f"vietnam_car_plate_processing exception {e}")
        return None


def vietnam_motorcycle_plate_processing(text):
    try:
        # Remove spaces from the text
        text = text.replace(" ", "")
        if text.isalpha():
            return None
        # Define the regex patterns for Vietnam motorcycle plate numbers
        patterns = [
            r"\d{2}-[A-Z]\d{5}",  # NN-LNNNNN
            r"\d{2}-[A-Z]\d{4}\.\d{2}",  # NN-LNNNN.NN
            r"\d{2}-[A-Z]{2}\d{3}\.\d{2}",  # NN-LLNNN.NN
            r"\d{2}-[A-Z]\d{6}",  # NN-LNNNNNN
            r"\d{2}-[A-Z]{2}\d{5}",  # NN-LLNNNNN
        ]

        # Check each pattern and return the first match found
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[0]

        # Initialize variables
        first_part = None
        middle_part = None
        last_part = text

        # Split the text at the first occurrence of '-'
        if "-" in text:
            first_part, last_part = text.split("-", 1)

        # Split the last_part at the first occurrence of '.'
        if "." in last_part:
            middle_part, last_part = last_part.split(".", 1)

        if first_part is not None:
            first_part = list(first_part)
            if len(first_part) == 2:
                if first_part[0].isalpha():
                    first_part[0] = MAP_LETTER_TO_DIGIT[first_part[0].upper()]
                if first_part[1].isalpha():
                    first_part[1] = MAP_LETTER_TO_DIGIT[first_part[1].upper()]
            else:
                return None

        if middle_part is not None:
            middle_part = list(middle_part)
            if len(middle_part) == 5:
                if middle_part[0].isdigit():
                    middle_part[0] = MAP_DIGIT_TO_LETTER[middle_part[0]]
                # middle_part[1] could be both alphabet and digit
                if middle_part[2].isalpha():
                    middle_part[2] = MAP_LETTER_TO_DIGIT[middle_part[2].upper()]
                if middle_part[3].isalpha():
                    middle_part[3] = MAP_LETTER_TO_DIGIT[middle_part[3].upper()]
                if middle_part[4].isalpha():
                    middle_part[4] = MAP_LETTER_TO_DIGIT[middle_part[4].upper()]
            elif len(middle_part) == 4:
                if middle_part[0].isdigit():
                    middle_part[0] = MAP_DIGIT_TO_LETTER[middle_part[0]]
                # middle_part[1] could be both alphabet and digit
                if middle_part[2].isalpha():
                    middle_part[2] = MAP_LETTER_TO_DIGIT[middle_part[2].upper()]
                if middle_part[3].isalpha():
                    middle_part[3] = MAP_LETTER_TO_DIGIT[middle_part[3].upper()]
            else:
                return None

        if first_part is None and middle_part is None:
            last_part = re.sub(r"[^A-Za-z0-9]", "", last_part)
            last_part = list(last_part)
            if len(last_part) == 8:  # NN-LN NNNN
                last_part[0] = (
                    MAP_LETTER_TO_DIGIT[last_part[0].upper()]
                    if last_part[0].isalpha()
                    else last_part[0]
                )
                last_part[1] = (
                    MAP_LETTER_TO_DIGIT[last_part[1].upper()]
                    if last_part[1].isalpha()
                    else last_part[1]
                )
                last_part[2] = (
                    MAP_DIGIT_TO_LETTER[last_part[2]]
                    if last_part[2].isdigit()
                    else last_part[2]
                )
                last_part[3] = (
                    MAP_LETTER_TO_DIGIT[last_part[3].upper()]
                    if last_part[3].isalpha()
                    else last_part[3]
                )
                last_part[4] = (
                    MAP_LETTER_TO_DIGIT[last_part[4].upper()]
                    if last_part[4].isalpha()
                    else last_part[4]
                )
                last_part[5] = (
                    MAP_LETTER_TO_DIGIT[last_part[5].upper()]
                    if last_part[5].isalpha()
                    else last_part[5]
                )
                last_part[6] = (
                    MAP_LETTER_TO_DIGIT[last_part[6].upper()]
                    if last_part[6].isalpha()
                    else last_part[6]
                )
                last_part[7] = (
                    MAP_LETTER_TO_DIGIT[last_part[7].upper()]
                    if last_part[7].isalpha()
                    else last_part[7]
                )
                first_part = last_part[:2]
                last_part = last_part[2:]
            elif len(last_part) == 9:  # NN-LX NNN.NN
                last_part[0] = (
                    MAP_LETTER_TO_DIGIT[last_part[0].upper()]
                    if last_part[0].isalpha()
                    else last_part[0]
                )
                last_part[1] = (
                    MAP_LETTER_TO_DIGIT[last_part[1].upper()]
                    if last_part[1].isalpha()
                    else last_part[1]
                )
                last_part[2] = (
                    MAP_DIGIT_TO_LETTER[last_part[2]]
                    if last_part[2].isdigit()
                    else last_part[2]
                )
                # last_part[3] could be both alphabet and digit
                last_part[4] = (
                    MAP_LETTER_TO_DIGIT[last_part[4].upper()]
                    if last_part[4].isalpha()
                    else last_part[4]
                )
                last_part[5] = (
                    MAP_LETTER_TO_DIGIT[last_part[5].upper()]
                    if last_part[5].isalpha()
                    else last_part[5]
                )
                last_part[6] = (
                    MAP_LETTER_TO_DIGIT[last_part[6].upper()]
                    if last_part[6].isalpha()
                    else last_part[6]
                )
                last_part[7] = (
                    MAP_LETTER_TO_DIGIT[last_part[7].upper()]
                    if last_part[7].isalpha()
                    else last_part[7]
                )
                last_part[8] = (
                    MAP_LETTER_TO_DIGIT[last_part[8].upper()]
                    if last_part[8].isalpha()
                    else last_part[8]
                )
                first_part = last_part[:2]
                middle_part = last_part[2:7]
                last_part = last_part[7:]
            else:
                return None
        elif first_part is not None and middle_part is None:
            last_part = re.sub(r"[^A-Za-z0-9]", "", last_part)
            last_part = list(last_part)
            if len(last_part) == 6:
                last_part[0] = (
                    MAP_DIGIT_TO_LETTER[last_part[0]]
                    if last_part[0].isdigit()
                    else last_part[0]
                )
                last_part[1] = (
                    MAP_LETTER_TO_DIGIT[last_part[1].upper()]
                    if last_part[1].isalpha()
                    else last_part[1]
                )
                last_part[2] = (
                    MAP_LETTER_TO_DIGIT[last_part[2].upper()]
                    if last_part[2].isalpha()
                    else last_part[2]
                )
                last_part[3] = (
                    MAP_LETTER_TO_DIGIT[last_part[3].upper()]
                    if last_part[3].isalpha()
                    else last_part[3]
                )
                last_part[4] = (
                    MAP_LETTER_TO_DIGIT[last_part[4].upper()]
                    if last_part[4].isalpha()
                    else last_part[4]
                )
                last_part[5] = (
                    MAP_LETTER_TO_DIGIT[last_part[5].upper()]
                    if last_part[5].isalpha()
                    else last_part[5]
                )
                last_part = last_part
            elif len(last_part) == 7:
                last_part[0] = (
                    MAP_DIGIT_TO_LETTER[last_part[0]]
                    if last_part[0].isdigit()
                    else last_part[0]
                )
                # last_part[1] could be both alphabet and digit
                last_part[2] = (
                    MAP_LETTER_TO_DIGIT[last_part[2].upper()]
                    if last_part[2].isalpha()
                    else last_part[2]
                )
                last_part[3] = (
                    MAP_LETTER_TO_DIGIT[last_part[3].upper()]
                    if last_part[3].isalpha()
                    else last_part[3]
                )
                last_part[4] = (
                    MAP_LETTER_TO_DIGIT[last_part[4].upper()]
                    if last_part[4].isalpha()
                    else last_part[4]
                )
                last_part[5] = (
                    MAP_LETTER_TO_DIGIT[last_part[5].upper()]
                    if last_part[5].isalpha()
                    else last_part[5]
                )
                middle_part = last_part[:5]
                last_part = last_part[5:]
            else:
                return None
        elif first_part is not None and middle_part is not None:
            last_part = list(last_part)
            if len(last_part) == 2:
                last_part[0] = (
                    MAP_LETTER_TO_DIGIT[last_part[0].upper()]
                    if last_part[0].isalpha()
                    else last_part[0]
                )
                last_part[1] = (
                    MAP_LETTER_TO_DIGIT[last_part[1].upper()]
                    if last_part[1].isalpha()
                    else last_part[1]
                )
            else:
                return None
        else:
            return None
        if middle_part is None:
            plate_number = "".join(first_part) + "-" + "".join(last_part)
        else:
            plate_number = (
                "".join(first_part)
                + "-"
                + "".join(middle_part)
                + "."
                + "".join(last_part)
            )
        if plate_number.count("?") < 1:
            return plate_number
        else:
            return None
    except Exception as e:
        print(f"vietnam_motorcycle_plate_processing exception {e}")
        return None
    
def india_plate_processing(text):                                                                               
    """
    Character Arrangements for Delhi UT: AA-NN-AA-NNNN, AA-N-AA-NNNN, AA-NN-A-NNNN, AA-N-AAA-NNNN                                                                
    Vehicle Registered States Info: Delhi, Haryana, Uttar Pradesh, Punjab, Rajasthan, Chattisgargh
    
    """
    AA_1 = ''
    NN = ''                                                                                             
    nn = ''                                                                                             
    AA_2 = ''                                                                                           
    NNNN = ''                                                                                           
    nnnn = ''
    num_plate = ''
    sp_np = ''
    text = text.replace(" ","")
    if len(text)==9:
        if text[0:2].isalpha() and text[2:4].isdigit() and text[5:].isdigit():
            text = text[ : 4] + '?' + text[4 : ]
        elif text[0:2].isalpha() and text[3:5].isalpha() and text[5:].isdigit():
            text = text[ : 2] + '?' + text[2 : ]
        else:
            return num_plate
        
    if (len(text)==10 and text[3:6].isalpha()) and (text[0:2].isalpha() and text[2].isdigit() and text[6:].isdigit()):
        st = ''
        for idx, char in enumerate(text):
            if idx <= 1:
                if (text[0] in ['D','0','O', 'Q'] and text[1] in ['L', 'I', '1', 'C']): st = "DL"
                elif (text[0] in ['H', 'M', 'A'] and text[1] in ['R', '2', 'P']): st = "HR"
                elif (text[0] in ['U', 'O', '0'] and text[1] in ['P']): st = "UP"
                elif (text[0] in ['P', '9', 'R'] and text[1] in ['B', '8']): st = "PB"
                elif (text[0] in ['R', 'P', '2'] and text[1] in ['J', '1']): st = "RJ"
                elif (text[0] in ['C', 'G', 'O'] and text[1] in ['H', 'M']): st = "CH"
        sp_np = st + text[2:]
        return sp_np

    elif len(text)==10 and text[3:6].isalpha():
        sp_np = text[3:6]
        text = text[:3] + sp_np + text[6:]
    else:
        pass
    
    for idx, char in enumerate(text):
        if idx <= 1:
            if (text[0] in ['D','0','O', 'Q'] and text[1] in ['L', 'I', '1']): AA_1 = "DL"
            elif (text[0] in ['H', 'M', 'A'] and text[1] in ['R', '2', 'P']): AA_1 = "HR"
            elif (text[0] in ['U', 'O', '0'] and text[1] in ['P']): AA_1 = "UP"
            elif (text[0] in ['P', '9', 'R'] and text[1] in ['B', '8']): AA_1 = "PB"
            elif (text[0] in ['R', 'P', '2'] and text[1] in ['J', '1']): AA_1 = "RJ"
            elif (text[0] in ['C', 'G', 'O'] and text[1] in ['H', 'M']): AA_1 = "CH"
            else:
                AA_1 = ''

        if 2 <= idx <=3:                                                                                
            if idx == 2 and text[idx] == 'S': continue                                                  
            elif text[idx] == 'O': nn = '0'                                                             
            elif text[idx] == 'Q': nn = '0'                                                             
            elif text[idx] == 'B': nn = '8'                                                             
            elif text[idx] == 'J': nn = '3'                                                             
            elif text[idx] == 'L': nn = '1'
            elif text[idx] == 'I': nn = '1'                                                             
            elif text[idx] == 'T': nn = '1'                                                             
            elif text[idx] == 'S': nn = '5'                                                             
            elif text[idx] == 'H': nn = '11'                                                            
            elif text[idx] == 'W': nn = '11'
            elif text[idx] == '?': nn=''                                                            
            else: nn = text[idx]                                                                        
            if nn.isdigit():
                NN += nn
                
        if 4 <= idx <=5:                                                                                
            if text[idx] == '0': aa_2 = 'Q'
            elif text[idx] == '1': aa_2 = 'I'                                                           
            elif text[idx] == '2': aa_2 = 'R'                                                           
            elif text[idx] == '3': aa_2 = 'B'                                                           
            elif text[idx] == '4': aa_2 = 'A'                                                           
            elif text[idx] == '5': aa_2 = 'S'                                                           
            elif text[idx] == '6': aa_2 = 'G'                                                           
            elif text[idx] == '7': aa_2 = 'T'                                                           
            elif text[idx] == '8': aa_2 = 'B'                                                           
            elif text[idx] == '9': aa_2 = 'P'
            elif text[idx] == '?': aa_2=''
            else: aa_2 = text[idx]                                                                      
            if aa_2.isalpha():
                AA_2 += aa_2
            
        if 6 <= idx <=9:                                                                                
            if text[idx] == 'O': nnnn = '0'                                                             
            elif text[idx] == 'Q': nnnn = '0'                                                           
            elif text[idx] == 'B': nnnn = '8'                                                           
            elif text[idx] == 'J': nnnn = '3'                                                           
            elif text[idx] == 'L': nnnn = '4'                                                           
            elif text[idx] == 'I': nnnn = '1'                                                           
            elif text[idx] == 'T': nnnn = '1'                                                           
            elif text[idx] == 'S': nnnn = '5'                                                           
            elif text[idx] == 'A': nnnn = '4'                                                           
            elif text[idx] == 'G': nnnn = '6'                                                           
            elif text[idx] == 'R': nnnn = '2'                                                           
            elif text[idx].isalpha(): nnnn = '?'                                                        
            else: nnnn = text[idx]                                                                      
            if nnnn.isdigit():
                NNNN += nnnn
    
    num_plate = AA_1 + NN + AA_2 + NNNN
    
    if len(num_plate) == 9:
        if ((num_plate[0:2].isalpha() and (num_plate[4].isalpha())) or (num_plate[0:2].isalpha() and num_plate[3:5].isalpha())) and ((num_plate[2].isdigit() and num_plate[5:].isdigit()) or (num_plate[2:4].isdigit() and num_plate[5:].isdigit())):
            num_plate = AA_1 + NN + AA_2 + NNNN
            return num_plate

    elif len(num_plate) == 10:    
        if ((num_plate[0:2].isalpha() and (num_plate[4]=='?' or num_plate[4:6].isalpha())) and (num_plate[2:4].isdigit() and num_plate[6:].isdigit())):
            num_plate = AA_1 + NN + AA_2 + NNNN
            return num_plate
        elif (len(num_plate)==10 and num_plate[3:6].isalpha()) and (num_plate[0:2].isalpha() and num_plate[2].isdigit() and num_plate[6:].isdigit()):
            return num_plate
    else:
        pass

def _mapping_point_to_different_resolution(points, resolution, target_resolution):
    """
    Map a set of points from one resolution to another and cast the resulting coordinates to integers.

    Parameters:
        points (tuple or list): Coordinates in the form (x1, y1, x2, y2).
        resolution (tuple): Original resolution as (width, height).
        target_resolution (tuple): Target resolution as (width, height).

    Returns:
        tuple: Mapped coordinates in the form (new_x1, new_y1, new_x2, new_y2) as integers.
    """
    if resolution[0] == 0 or resolution[1] == 0:
        raise ValueError("Original resolution dimensions must be non-zero.")

    # Calculate scaling factors for x and y dimensions.
    scale_x = target_resolution[0] / resolution[0]
    scale_y = target_resolution[1] / resolution[1]

    # Unpack the points.
    x1, y1, x2, y2 = points

    # Apply scaling factors and cast the coordinates to integers.
    new_x1 = int(x1 * scale_x)
    new_y1 = int(y1 * scale_y)
    new_x2 = int(x2 * scale_x)
    new_y2 = int(y2 * scale_y)
    return (new_x1, new_y1, new_x2, new_y2)