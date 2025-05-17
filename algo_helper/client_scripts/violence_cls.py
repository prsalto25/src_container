import numpy as np
import zmq
import cv2
cv2.setNumThreads(1)

from .pose_estimation import PoseEstimator
from ..use.preprocessing import generate_pairs, is_duplicate

import socket

def _check_port_active(host: str, port: int) -> bool:
    """Check if a port is active on a given host.
    - ex usage: 
    _flag_server_is_on = _check_port_active("localhost", int("2687"))
    print(f"Pose Estimator server is active : {_flag_server_is_on}")
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2)  # Set a timeout for faster response
        result = s.connect_ex((host, port))
        return result == 0  # Returns True if the port is open

class ActionClassifier:
    # def __init__(self, docker_url='tcp://127.0.0.1:2562'):
    def __init__(self, docker_url='tcp://127.0.0.1:2197'):
        #2557
        _flag_server_is_on = _check_port_active("localhost", int("2197"))
        print(f"ActionClassifier server is active : {_flag_server_is_on}")
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(docker_url)
        # self.class_dict = {0:"lyingdown", 1:"standing1", 2:"standing2", 3:"sitting", 4:"standing0"}
        self.class_dict = {0:"non_violence", 1:"violence"}
        self.pose_estimator = PoseEstimator()
    
    def offset_bbox_coordinates(self, bboxes, offset):

       # Offset the bbox coordinates
        bboxes[:, [0, 2]] -= offset[0]
        bboxes[:, [1, 3]] -= offset[1]

        return bboxes
    
    def offset_pose_results(self, poses, offset):

       # Offset the bbox coordinates
        poses[:, :, 0] -= offset[0]
        poses[:, :, 1] -= offset[1]

        return poses
    
    def undo_offsets(self, pose_results, offsets):
        # Add axis in between to make it broadcastable
        offsets_expanded = np.expand_dims(np.expand_dims(offsets, axis=1), axis=1)
        pose_results[...,:2] += offsets_expanded
        return pose_results

    def offset_pose_coordinates(self, bboxes, pose_coords):
        offset_pose_coords = []

        for bbox, pose_coord in zip(bboxes, pose_coords):
            x_min, y_min, x_max, y_max = bbox
            pose_coords_bbox = pose_coord.copy()  # Make a copy to avoid modifying the original array

            # Calculate the offset
            x_offset = x_min
            y_offset = y_min

            # Offset the pose coordinates for each keypoint
            pose_coords_bbox[:, 0] += x_offset
            pose_coords_bbox[:, 1] += y_offset

            # pose_coords_bbox[:, 0] = x_max - pose_coords_bbox[:, 0]
            # pose_coords_bbox[:, 1] += y_max - pose_coords_bbox[:, 0]

            offset_pose_coords.append(pose_coords_bbox)

        return np.array(offset_pose_coords)

    def get_best_class(self, class_result):
        if class_result.any():
            class_id = np.argmax(class_result)
            prob = class_result[class_id]
            class_name = self.class_dict[class_id]
        else:
            class_name = "other"
            prob = 0

        return class_name, prob

    def batch_inference(self, data, batch_size=64):
        # Initialize an empty list to store batch results
        all_batch_results = []
        
        # Calculate the number of batches
        num_batches = int(np.ceil(len(data) / batch_size))
        
        # Process each batch
        for i in range(num_batches):
            # Get the start and end indices for the current batch
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(data))
            
            # Extract the batch from the data
            batch_data = data[start_idx:end_idx]
            
            # Send the batch for inference
            self.socket.send_pyobj(batch_data)
            
            # Receive the results for the current batch
            batch_results = self.socket.recv_pyobj().reshape(-1, 2)
            
            # Append the batch results to the list
            all_batch_results.append(batch_results)
        
        # Concatenate all batch results into a single array
        final_results = np.concatenate(all_batch_results, axis=0)
        
        # Return the final concatenated results
        return final_results
    
    def get_poses(self, frame, bboxes, margin=0.05):
        pose_results = []
        for box in bboxes:
            # We send a bigger crop for better pose results
            margin_box = box.copy()
            d = int(margin * np.max(box[:4]))

            margin_box[[0, 1]] -= d

            # We have to take into account negative dx, dy when removing the
            # margins later.
            dx, dy = margin_box[[0, 1]]
            margin_box[[2, 3]] += d

            # If it resulted in negative value, we have to know the actual
            # offsets after clipping. For ex. if x1,y1 becomes -10,-10 after
            # adding margin above and d is 20, then after clipping, x1,y1
            # becomes 0,0. So the actual offset is 10,10, not 20,20.
            dx = d if not dx < 0 else d + dx
            dy = d if not dy < 0 else d + dy

            # Clip to bounds
            margin_box = np.where(margin_box < 0, 0, margin_box)

            # Offset box relative to margin. Used by RTMPose to return pose
            # results only for the person inside this box. Since we added
            # margin, we have to tell RTMPose where the person is, otherwise it
            # will include margin.

            offset_box = box[:4].copy()
            offset_box[[2, 3]] -= (offset_box[[0, 1]] - (dx, dy))
            offset_box[[0, 1]] = (dx, dy)

            # Clip to bounds
            offset_box = np.where(offset_box < 0, 0, offset_box)


            # Crop using margin box
            x1, y1, x2, y2, _, _ = margin_box
            frame_person = frame[y1:y2, x1:x2]

            # Get pose result. Return result is relative to margin.
            result = self.pose_estimator.detect(frame_person, offset_box)

            # Remove margin for pose coordinates
            result[:, :, 0] -= dx
            result[:, :, 1] -= dy

            # Offset pose coordinates so that it is relative to the whole
            # image and not just the cropped region sent to RTMPose
            result = self.offset_pose_coordinates([box[:4]], result)

            pose_results += result.tolist()
        return pose_results
    
    def get_low_conf_idxs(self, poses):
        poses = np.array(poses)


    def detect_single(self, frame, bbox_pairs, tid_pairs, all_boxes):
        # Run pose detection on all the individual bboxes
        # bboxes = [pair[0] for pair in bbox_pairs]

        poses = self.get_poses(frame, all_boxes)

        # Turn into a dict so that we can access each pose by their bbox position no.
        poses = {tid:pose for tid, pose in zip(all_boxes[:, 5], poses)}

        # Generate pairs. We get a list containing the pair and the merged bbox
        # and their positions from the pair [[bbox1, bbox2], merged_bbox,
        # [bbox1 pos, bbox2 pos]]
        paired_bboxes = generate_pairs(bbox_pairs, tid_pairs)

        # Process pose results
        # We have to apply offsets
        pose_results = []
        paired_poses = []
        offsets = []
        paired_bboxes_filt = []
        for i, (bbox_pair, merged_box, tid_pair) in enumerate(paired_bboxes):
            kp1 = np.array(poses[tid_pair[0]])
            kp2 = np.array(poses[tid_pair[1]])
            
            # Check if they're duplicates
            if is_duplicate(kp1, kp2):
                mask = False
                continue

            # Check if they are low confidence
            if kp1[:, -1].mean() < 0.05 or kp2[:, -1].mean() < 0.05:
                continue

            x1, y1, x2, y2 = merged_box

            # Offset coordinates based on merged bbox, so that the bboxes
            # correspond with the cropped frame_person. Store the offset for
            # undoing it later.
            offsets += [(x1, y1)]
            paired_poses = (kp1, kp2)
            pose_result = self.offset_pose_results(np.array(paired_poses), (x1, y1))
            
            # Pose results of both bboxes; they're paired too.
            # We obtain it from the already inferred results
            pose_results += pose_result.tolist()
            paired_bboxes_filt += [(bbox_pair, merged_box, tid_pair)]
        
        # print(pose_results)
        # frame = cv2.rectangle(frame, paired_bboxes[0][1][:2], paired_bboxes[0][1][2:], (255,0,0), 2)
        # for point in kp1[:,:2]:
        #     frame = cv2.circle(frame, point.astype(int), 5, (0, 0, 255), -1)
        # cv2.imwrite("output.jpg", frame)
        # import pdb
        # pdb.set_trace()
    
        # Reshape pose results.
        pose_results = np.array(pose_results).astype(np.float64).reshape(-1, 2, 17, 3)

        if pose_results.size:
            # 2. Send to classifier
            # Maximum 64 at a time
            class_results = self.batch_inference(pose_results)
        
            # Undo offsets
            paired_poses = self.undo_offsets(pose_results.copy(), np.array(offsets))
        else:
            class_results = np.zeros(2).reshape(-1, 2)
            paired_poses = np.empty_like(pose_results)
        # print(pose_results)
        # print(class_results)
        
        return list(zip(class_results, pose_results, paired_poses, paired_bboxes_filt))

    def detect(self, frame, bbox_pairs, tid_pairs):
        # Generate pairs. We get a list containing the pair and the merged bbox
        paired_bboxes = np.array(generate_pairs(bbox_pairs, tid_pairs))

        # 1. Send to pose
        pose_results = []
        offsets = []
        for bbox_pair, merged_box, tid_pair in paired_bboxes:
            x1, y1, x2, y2 = merged_box
            frame_person = frame[y1:y2, x1:x2]

            # Offset coordinates based on merged bbox, so that the bboxes
            # correspond with the cropped frame_person. Store the offset for
            # undoing it later.
            offsets += [(x1, y1)]
            bbox_pair = self.offset_bbox_coordinates(np.array(bbox_pair), (x1, y1))[:, :4]
            
            # Pose results of both bboxes; they're paired too.
            pose_results += self.pose_estimator.detect(frame_person, bbox_pair).tolist()
        
        # print(pose_results)
        
        # Reshape pose results.
        pose_results = np.array(pose_results).astype(np.float32).reshape(-1, 2, 17, 3)

        if pose_results.size:
            # 2. Send to classifier
            self.socket.send_pyobj(pose_results)
            class_results = self.socket.recv_pyobj().reshape(-1, 2)
        
            # Undo offsets
            paired_poses = self.undo_offsets(pose_results.copy(), np.array(offsets))
        else:
            class_results = np.zeros(2).reshape(-1, 2)
            paired_poses = np.empty_like(pose_results)
        
        return list(zip(class_results, pose_results, paired_poses, paired_bboxes))