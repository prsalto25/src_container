import numpy as np
import cv2

from collections import deque


class StateTracker:
    def __init__(self, trackid):
        self.trackid = trackid
        # Initialize a buffer to store the last 30 bounding box centroids (x, y)
        self.buffer_size = 4
        self.centroid_buffer = deque(maxlen=self.buffer_size)
        self.frame_index = 0

        # Initialize variables for calculating the average direction and magnitude
        self.average_direction = None
        self.average_magnitude = 0
        self.angle = None

        # Initialize variables for height rate calculation
        self.height_buffer = deque(maxlen=self.buffer_size)
        self.height_rate = 0

        # Initialize variables for elbow motion
        self.elbow_angle_buffer = {
            "left": deque(maxlen=self.buffer_size),
            "right": deque(maxlen=self.buffer_size),
        }
        self.average_elbow_angle_change = [0, 0]

    # Function to calculate the centroid of a bounding box
    def calculate_centroid(self, bounding_box):
        x1, y1, x2, y2 = bounding_box
        centroid_x = (x1 + x2) / 2
        centroid_y = (y1 + y2) / 2
        return (centroid_x, centroid_y)

    # Function to calculate the average direction and magnitude of the centroid vectors
    def calculate_average_direction_magnitude(self):
        # Calculate the vectors between consecutive centroids
        v1 = [self.centroid_buffer[i] for i in range(0, self.buffer_size - 1)]
        v2 = [self.centroid_buffer[i] for i in range(1, self.buffer_size)]
        vectors = np.array(v2) - np.array(v1)

        # Calculate the sum of vectors
        sum_vectors = np.sum(vectors, axis=0)

        # Calculate the average direction (normalized) and magnitude
        self.average_magnitude = np.linalg.norm(sum_vectors)
        self.average_direction = sum_vectors / (self.average_magnitude + 1e-6)

        # Calculate the angle (in degrees) from the direction vector
        self.angle = np.degrees(
            np.arctan2(self.average_direction[1], self.average_direction[0])
        )

    # Function to calculate the rate of change of height
    def update_height_rate(self, height):
        # Calculate delta between consecutive heights
        h1 = [self.height_buffer[i] for i in range(0, self.buffer_size - 1)]
        h2 = [self.height_buffer[i] for i in range(1, self.buffer_size)]
        delta = np.array(h2) - np.array(h1)

        # Calculate the rate of change
        if len(self.height_buffer) >= self.buffer_size:
            self.height_rate = np.mean(delta)
        else:
            self.height_rate = None

    def update_state(self, bounding_box):
        # Calculate the centroid of the bounding box
        centroid = self.calculate_centroid(bounding_box)

        # Update the centroid buffer
        self.centroid_buffer.append(centroid)

        # Update height buffer
        y1, y2 = bounding_box[1], bounding_box[3]
        self.height_buffer.append(y2 - y1)

        if (
            len(self.centroid_buffer) >= self.buffer_size
            and len(self.height_buffer) >= self.buffer_size
        ):
            # Update vector metric
            self.calculate_average_direction_magnitude()
            self.update_height_rate(y2 - y1)

    # Function to calculate the angle given three points
    def calculate_angle(self, shoulder, elbow, wrist):
        vector1 = np.array([shoulder[0] - elbow[0], shoulder[1] - elbow[1]])
        vector2 = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]])

        dot_product = np.dot(vector1, vector2)
        magnitude_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)

        if magnitude_product == 0:
            return 0  # Avoid division by zero

        cos_angle = dot_product / magnitude_product
        angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    # Function to update elbow motion
    def update_elbow_motion(
        self,
        left_shoulder,
        left_elbow,
        left_wrist,
        right_shoulder,
        right_elbow,
        right_wrist,
    ):
        # Calculate left elbow angle
        left_current_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)

        # Update left elbow angle buffer
        self.elbow_angle_buffer["left"].append(left_current_angle)

        # Calculate right elbow angle
        right_current_angle = self.calculate_angle(
            right_shoulder, right_elbow, right_wrist
        )

        # Update right elbow angle buffer
        self.elbow_angle_buffer["right"].append(right_current_angle)

        if len(self.elbow_angle_buffer["left"]) >= self.buffer_size:
            # Calculate absolute average change in left elbow angle
            left_angle_changes = [
                abs(
                    self.elbow_angle_buffer["left"][i]
                    - self.elbow_angle_buffer["left"][i - 1]
                )
                for i in range(1, len(self.elbow_angle_buffer["left"]))
            ]
            self.average_elbow_angle_change[0] = np.mean(left_angle_changes)

        if len(self.elbow_angle_buffer["right"]) >= self.buffer_size:
            # Calculate absolute average change in right elbow angle
            right_angle_changes = [
                abs(
                    self.elbow_angle_buffer["right"][i]
                    - self.elbow_angle_buffer["right"][i - 1]
                )
                for i in range(1, len(self.elbow_angle_buffer["right"]))
            ]
            self.average_elbow_angle_change[1] = np.mean(right_angle_changes)

    def vector_intersects_bbox(self, bounding_box):
        if self.average_direction is None:
            return False

        cx, cy = self.centroid_buffer[-1]
        x1, y1, x2, y2 = bounding_box

        # Calculate intersection points with the four sides of the bounding box
        t_left = (x1 - cx) / self.average_direction[0]
        t_right = (x2 - cx) / self.average_direction[0]
        t_top = (y1 - cy) / self.average_direction[1]
        t_bottom = (y2 - cy) / self.average_direction[1]

        # Determine the valid range of t values for intersection
        t_min = min(t_left, t_right)
        t_max = max(t_left, t_right)
        t_min = max(t_min, min(t_top, t_bottom))
        t_max = min(t_max, max(t_top, t_bottom))

        # Check if there is a valid intersection within the bounding box
        return t_min <= t_max and t_min >= 0  # Ensure t_min is non-negative

    def visualize_vector(self, frame):
        if self.average_direction is not None:
            # Centroid
            centroid_x, centroid_y = self.centroid_buffer[-1]
            centroid = (int(centroid_x), int(centroid_y))

            # Scale the direction vector by its magnitude
            scaled_direction = self.average_direction * self.average_magnitude

            # Calculate the endpoint of the vector by adding the scaled direction to the centroid
            endpoint = (
                int(centroid_x + scaled_direction[0]),
                int(centroid_y + scaled_direction[1]),
            )

            # Draw the scaled direction vector (blue arrow)
            cv2.arrowedLine(frame, centroid, endpoint, (0, 0, 255), 2, tipLength=0.2)
