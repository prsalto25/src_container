import numpy as np
from powerboxes import iou_distance

def generate_pairs(bbox_pairs, tid_pairs):
    # List containing the merged bboxes through union
    merged_bboxes = []
    paired_bboxes = []
    for box_pair, tid_pair in zip(bbox_pairs, tid_pairs):
        # Pairs are stored as [bbox1, [list of bbox pairs of bbox1]]
        bbox1 = box_pair[0]
        tid1 = tid_pair[0]
        bbox_list = box_pair[1]
        tid_list = tid_pair[1]
        for bbox2, tid2 in zip(bbox_list, tid_list):
            # Safeguard: Avoid pairing with the same box
            if tid1 == tid2:
                continue
            # Avoid duplicates
            merged_box = union_bboxes(bbox1, bbox2)
            if not bbox_has_high_iou(merged_box, merged_bboxes):
                merged_bboxes.append(merged_box)
                paired_bboxes.append((np.array((bbox1, bbox2)), merged_box, np.array((tid1, tid2))))

    return paired_bboxes

def bbox_has_high_iou(query_bbox, bbox_list, iou_thresh=0.95):
    if not bbox_list:
        return False

    iou = 1- iou_distance(np.array([query_bbox[:4]]), np.array(bbox_list[:4]))[0]

    # Check if the intersection area with any bounding box is greater than the threshold
    return np.any(iou > iou_thresh)

def union_bboxes( bbox1, bbox2):
    # Creates a union bbox of bbox1 and bbox2
    x1 = min(bbox1[0], bbox2[0])
    y1 = min(bbox1[1], bbox2[1])
    x2 = max(bbox1[2], bbox2[2])
    y2 = max(bbox1[3], bbox2[3])
    return [x1, y1, x2, y2]

def is_duplicate(kp1, kp2, threshold=10):
    # Calculate the Euclidean distance between corresponding keypoints
    distances = np.linalg.norm(np.array(kp1) - np.array(kp2), axis=1)

    # Calculate the average distance
    average_distance = np.mean(distances)

    # Determine if they are duplicates based on a threshold
    return average_distance < threshold