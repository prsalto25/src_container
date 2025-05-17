import zmq
import numpy as np
import cv2

class CarInfoZMQ:
    def __init__(self, docker_url='tcp://localhost:3170'):
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(docker_url)
        
        self.brand_dict = {
            0: 'Ashok Leyland', 1: 'Audi', 2: 'Bentley', 3: 'Bharat Benz',
            4: 'BMW', 5: 'Eicher Motors', 6: 'Ford', 7: 'Honda', 8: 'Hyundai',
            9: 'Jaguar', 10: 'KIA', 11: 'Land Rover', 12: 'Mahindra',
            13: 'Maruti Suzuki', 14: 'Mercedes', 15: 'MG Motors', 16: 'Nissan',
            17: 'Renault', 18: 'Rolls Royce', 19: 'Skoda', 20: 'Swaraj Mazda',
            21: 'Tata', 22: 'Toyota', 23: 'Volkswagen', 24: 'Volvo'
        }
        
        self.color_dict = {
            0: 'black', 1: 'white', 2: 'silver', 3: 'gray', 4: 'red',
            5: 'red2', 6: 'orange', 7: 'yellow', 8: 'green', 9: 'blue',
            10: 'purple', 11: 'brown', -1: 'unknown'
        }

        self.class_names = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck',
            8: 'boat',
        }

        # Different colors for different vehicle types
        self.colors = {
            2: (0, 255, 0),  # car: green
            3: (255, 0, 255),  # motorcycle: magenta
            5: (0, 165, 255),  # bus: orange
            7: (0, 0, 255),  # truck: red
            8: (255, 255, 0)  # boat: cyan
        }
    
    def detect_vehicles(self, frame):
        if frame is None or frame.size == 0:
            print("Invalid frame")
            return []
            
        frame = frame.astype(np.uint8)
        h, w, c = frame.shape
        frame_shape = f"{h}_{w}_{c}"
        
        _, encoded_frame = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        frame_bytes = encoded_frame.tobytes()
        
        print("Sending image for vehicle and brand detection...")
        self.socket.send_multipart([frame_bytes, frame_shape.encode()])
        
        result_bytes = self.socket.recv()
        print("Received detection results")
        
        detections = []
        try:
            results = np.frombuffer(result_bytes, dtype=np.float32)
            
            if results.size > 0:
                results = results.reshape(-1, 3)
                
                for result in results:
                    brand_id = int(result[0])
                    brand_conf = float(result[1])
                    color_id = int(result[2])
                    
                    # Handle no-brand case
                    if brand_id == -1:
                        detections.append({
                            'color_id': color_id,
                            'color_name': self.color_dict.get(color_id, 'Unknown')
                        })
                    else:
                        detections.append({
                            'brand_id': brand_id,
                            'brand_name': self.brand_dict.get(brand_id, 'Unknown'),
                            'brand_conf': brand_conf,
                            'color_id': color_id,
                            'color_name': self.color_dict.get(color_id, 'Unknown')
                        })

        except Exception as e:
            print(f"Error parsing detection results: {e}")
            
        return detections

    def run(self, frame, yolo_results, plate_x1, plate_y1, plate_x2, plate_y2):
        # Parse detection
        vehicle_color_name = "unknown"
        vehicle_brand_name = "unknown"

        try:
            arr_box, arr_cls, arr_conf = [], [], []

            # Check if yolo_results is a list or a dictionary
            if isinstance(yolo_results, dict):
                vehicle_results = yolo_results.get("vehicle", [])
            else:
                vehicle_results = yolo_results

            for result in vehicle_results:
                if hasattr(result, 'attr'):
                    arr_box.append(result.attr.get("xyxy"))
                    arr_cls.append(result.attr.get("cls"))
                    arr_conf.append(result.attr.get("conf"))
                elif isinstance(result, dict):
                    arr_box.append(result.get("xyxy"))
                    arr_cls.append(result.get("cls"))
                    arr_conf.append(result.get("conf"))
        except Exception as e:
            print(f"VehicleColorAnalytic run error {e}")
            return frame, "unknown", "unknown"

        # Visualization
        for (box, class_id, box_score) in zip(arr_box, arr_cls, arr_conf):
            class_id = list(filter(lambda key: self.class_names[key] == class_id, self.class_names))
            class_id = class_id[0]
            # class_id = int(class_name)
            x1, y1, x2, y2 = box

            # if not (x1 < plate_x1 < x2 and y1 < plate_y1 < y2):
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
            # else:
            color = self.colors.get(class_id, (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if class_id in [2, 3, 5, 7, 8]:  # Vehicle classes

                vehicle_color_name = "N/A"
                # vehicle_color_bgr = None

                # Call detect_vehicles to get brand information
                vehicle_detections = self.detect_vehicles(frame)
                if vehicle_detections:
                    vehicle_brand_name = vehicle_detections[0].get("brand_name", "Unknown")
                    vehicle_color_name = vehicle_detections[0].get("color_name", "Unknown")

                class_name = self.class_names[class_id]
                label = f"{class_name}: {box_score:.2f}"
                if vehicle_color_name != "N/A":
                    label += f" ({vehicle_color_name})"
                if vehicle_brand_name != "unknown":
                    label += f" [{vehicle_brand_name}]"

                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                # if vehicle_color_bgr is not None:
                #     color_sample_size = 15
                #     cv2.rectangle(frame, (x1 + label_size[0] + 5, y1 - color_sample_size - 5),
                #                 (x1 + label_size[0] + 5 + color_sample_size, y1 - 5),
                #                 vehicle_color_bgr, -1)
            else:
                class_name = self.class_names.get(class_id, f"class_{class_id}")
                label = f"{class_name}: {box_score:.2f}"
                cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        return frame, vehicle_color_name, vehicle_brand_name
