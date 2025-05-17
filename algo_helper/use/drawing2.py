import cv2
import numpy as np
from typing import Tuple


def draw_roi(frame, pts, color):
    for i in range(len(pts)-1):
        cv2.line(frame, tuple(pts[i]), tuple(pts[i+1]), color,2)
    cv2.line(frame, tuple(pts[-1]), tuple(pts[0]), color,2)

def draw_rois(frame, rois, color):
    for roi in rois:
        draw_roi(frame, roi, color)


def draw_texted_bbox(
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