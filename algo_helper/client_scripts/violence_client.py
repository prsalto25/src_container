import cv2
cv2.setNumThreads(1)
import requests
import base64
import numpy as np

class ViolenceClsVLMKelvin:
    def __init__(self, server_url: str="http://localhost:18000/kelvin-vm7/classify-violence"):
        self.server_url = server_url

    def _encode_image_to_base64(self, image: np.ndarray) -> str:
        _, buffer = cv2.imencode('.jpg', image)  # Encode image as JPEG
        return base64.b64encode(buffer).decode('utf-8')

    def classify_vio(self, image: np.ndarray) -> dict:
        try:
            # Resize the image for consistency
            image_resized = cv2.resize(image, (960, 720))
            encoded_image = self._encode_image_to_base64(image_resized)

            # Prepare the payload
            payload = {
                "image_base64": encoded_image,
            }

            # Send the POST request to the FastAPI server
            response = requests.post(self.server_url, json=payload)

            # Check if the request was successful
            if response.status_code == 200:
                response_json = response.json()
                return response_json
            else:
                raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

        except Exception as e:
            print(f"Error: {str(e)}")
            return {"error": str(e)}

if __name__ == "__main__":
    vio_cls_vlm = ViolenceClsVLMKelvin()

    image_path = "img/8.png"
    image = cv2.imread(image_path)

    if image is not None:
        response_vio = vio_cls_vlm.classify_vio(image)
        try:
            if response_vio["flag_violence"] == True:
                print("True Positive alert: Send Alert")
            else:
                print("False Positive Alert: Ignore this alert")
        except:
            print("Internal error, please tell kelvin and just use this alert as true positive for temporal")
    else:
        print("Failed to load image. Check the image path.")





"""
class ViolenceVA:
    def __init__(self):
        self.lock = threading.Lock()  # Ensures thread safety

    USAGE ON OUR VA ! 
    def run_alert_in_thread(frame, confidence):
        alert_thread = threading.Thread(target=send_alert, args=(frame, confidence))
        alert_thread.start()

    def send_alert(self, frame, confidence): # need to write
        global prev_time
        date = self.timer.now.strftime('%Y-%m-%d %H:%M:%S')

        ## ----------------------------------------------------------------------------------------
        ## MODIFICATION IS HERE !!!!!!!!!!!!!!
        try:
            response_vio = vio_cls_vlm.classify_vio(frame)
            if response_vio["flag_violence"] == False:
                return 
        except: 
            print("HAVE INTERNAL ERROR !! PLEASE CONTACT KELVIN !")
        self.lock = threading.Lock()  # Ensures thread safety
        ## ----------------------------------------------------------------------------------------
            
            if abs(int(date[-2:]) - int(prev_time[-2:])) >= 5:
                uuid_ = str(uuid.uuid4())
                clip_path = 'None'
                mysql_values = [date,clip_path, confidence, self.attr['camera_name'], self.attr['camera_id'],
                        self.attr['id_branch'], self.attr['id_account'], uuid_, 2,"None"]
                r=self.mysql_helper.insert_fast('violence', mysql_values)
                date = self.timer.now.strftime('%Y-%-m-%-d_%H:%M:%S')
                date2 = date.replace(" ","_",1)
                imgName = f"/resources/{self.attr['id_account']}/{self.attr['id_branch']}/violence/{self.attr['camera_id']}/{date}_{uuid_}.jpg"
                imgPath = '/home' + imgName
                mysql_values2 = (uuid_, 'Violence Detection', date, date, 'NULL', self.attr['id_account'], self.attr['id_branch'], 2, self.attr['camera_name'], self.attr['camera_id'], imgPath, 'NULL', 'NULL')
                self.mysql_helper.insert_fast('tickets', mysql_values2)
                self.last_sent = self.timer.now_t
                date = self.timer.now.strftime('%Y-%m-%d %H:%M:%S')
                imgName = self.send_img(frame, date2, uuid_)
                videoName = self.send_video(frame,date2, uuid_) ## FOR VIDEO ALERT
                prev_time = self.timer.now.strftime('%Y-%-m-%-d_%H:%M:%S')

    def send_alert_new(self, frame, confidence):
        alert_thread = threading.Thread(target=self.send_alert, args=(frame, confidence))
        alert_thread.start()

    # CALL "send_alert_new" TO REPLACE THE "send_alert"  
"""

   
