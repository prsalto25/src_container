import zmq
import cv2
import numpy as np
import ast
import glob
import os
from datetime import datetime

class FaceDetectionClient:
    def __init__(self, docker_url='tcp://localhost:5603', shape=(1280, 720)):
        # Set up ZeroMQ context and socket
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(docker_url)
        self.shape = shape

        # Define categories for age, gender, emotion
        self.classLists = {
                'gender': ['Male', 'Female'],
                'age': ['Teen', 'Young Adult', 'Young Adult', 'Adult', 'Adult', 'Middle Age', 'Senior'],
                'emotion': ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'] 
                }

    def format_results(self, arr_results):
        formatted_dets = []
        for conf, xyxy, age, gender, emotion, facepoint, feature in arr_results:
            age = self.classLists["age"][age.index(max(age))]
            gender = self.classLists["gender"][gender.index(max(gender))]
            emotion = self.classLists["emotion"][emotion.index(max(emotion))]
            det = {
                    'conf': conf,
                    'xyxy': xyxy,                    
                    'facepoint': facepoint,
                    'age': age,
                    'gender': gender,
                    'emotion': emotion,
                    'feature': feature
                    }
                    
            formatted_dets.append(det)
            
        return formatted_dets

    def detect_faces(self, frame):
        frame = frame.astype(np.uint8)
        frame_shape = frame.shape
        text_frame_shape = f"{frame_shape[0]}_{frame_shape[1]}_{frame_shape[2]}"
        self.socket.send_multipart([frame, text_frame_shape.encode()])
        
        # Receive the detection results
        detection = self.socket.recv()
        # print(detection.decode())
        arr_results = ast.literal_eval(detection.decode())
        
        # Format the results
        try:
            formatted_dets = self.format_results(arr_results)
        except:
            formatted_dets = []
        return formatted_dets


class FaceRecognitionManagement:
    """
    Manages face recognition by loading a database of features and matching new features.
    """

    def __init__(self):
        self.arr_features = None
        self.database_fr_names = None
        self.user_name_list = None

    def load_database(self, folder_path, output_prefix="_k_arcface_ires100.npy"):
        arr_features = []
        database_fr_names = {}
        user_name_list = set()
        feature_idx_count = 0 

        # Search for all .npy files matching the given prefix
        print("LOAD FR DATABASE")
        user_dirs = glob.glob(folder_path + "*")
        print(user_dirs)
        for user_dir in user_dirs:
            user_name = os.path.basename(user_dir)
            print("Name: ", user_name)
            
            # Load all features db (.npy)
            feature_paths = glob.glob(user_dir + "/*" + output_prefix)
            for f_path in feature_paths: 
                print("Load feature: ", f_path)
                feature = np.load(f_path)
                # Store Data
                arr_features.append(feature)
                database_fr_names[feature_idx_count] = {"name": user_name}
                feature_idx_count += 1 
                user_name_list.add(user_name)

                assert len(arr_features) == feature_idx_count, (
                    f"Mismatch: len(arr_features)={len(arr_features)} vs feature_idx_count={feature_idx_count}"
                )
        self.arr_features = np.array(arr_features)
        print(self.arr_features.shape)
        self.arr_features = self.arr_features.squeeze(1)
        self.database_fr_names = database_fr_names
        self.user_name_list = user_name_list

        print("Loaded Database FR: ", len(user_dirs))
        print("Name list: ", user_name_list)

    def get_name(self, feature, thres=0.1):
        conf = None
        name = "Unknown"

        # Step #1: Check if the database is loaded
        if self.arr_features is None or len(self.arr_features) == 0:
            print("No database loaded.")
            return name

        # Step #2: Normalize the input feature into a numpy array
        feature = np.array([feature])
        feature = feature.reshape(1, 512)

        # Step #3: Compute cosine similarity between the feature and database
        similar = np.einsum('ij,kj->ik', feature, self.arr_features)
        max_score = np.amax(similar, axis=1)

        # Step #4: Compare similarity score against the threshold
        if max_score > thres:
            max_ind = np.where(similar == max_score[:, None])[1][0]

            # Step #5: Retrieve metadata for the matched feature
            personAtt = self.database_fr_names[max_ind]
            name = personAtt['name']
            conf = max_score[0]

        # Step #6: Return the matched name and ID (or None if no match)
        return name


class FRCreateDatabase:
    """
    Create Database for FR
        Folder format: 

        |database_fr
        |- <name> 
        |-- anyphoto.jpg
        |- <name> 
        |-- anyphoto.jpg
    """
    def __init__(self, fd_client: FaceDetectionClient):
        self.fd_client = fd_client

    def create_database(self, images_folder_path="./database_fr", output_prefix="_k_arcface_ires100.npy"):
              # 1. Create the main output directory
        today = datetime.now()
        output_folder = f"{images_folder_path}_npy_{today.day}_{today.month}_{today.year}"
        os.makedirs(output_folder, exist_ok=True)
        print(f"Output folder created at: {output_folder}")

        # 2. Load Images in Folders
        image_paths = glob.glob(images_folder_path + "/*/*.jpg") + \
                      glob.glob(images_folder_path + "/*/*.JPG") + \
                      glob.glob(images_folder_path + "/*/*.png") + \
                      glob.glob(images_folder_path + "/*/*.PNG")

        print(image_paths)
        # 3. Process Each Image
        for img_path in image_paths:
            print(f"Processing {img_path}")
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Could not read image: {img_path}")
                continue

            try:
                print(f"Image shape: {frame.shape}")
            except Exception as e:
                print(f"Image shape error for {img_path}: {e}")
                continue
            
            # 4. Detect Faces and Extract Features
            result = self.fd_client.detect_faces(frame)
            if len(result) == 0:
                print(f"No face detected in {img_path}")
                continue
            
            # 5. Create Subfolder for Base Name
            base_folder_name = img_path.split('/')[-2]
            subfolder_path = os.path.join(output_folder, base_folder_name)
            os.makedirs(subfolder_path, exist_ok=True)

            # 6. Save the Feature in the Subfolder
            base_name = os.path.basename(img_path).split('.')[0]
            npy_path_out = os.path.join(subfolder_path, base_name + output_prefix)
            np.save(npy_path_out, result[0]["feature"])
            print(f"Saved feature to {npy_path_out}")






# End of Library




















def process_image(image, fd_client, fr_manager):
    # Detect faces
    formatted_dets = fd_client.detect_faces(image)

    # Process each detected face and draw bounding boxes, labels, and face points
    for det in formatted_dets:
        conf = det['conf']
        xyxy = det['xyxy']
        age = det['age']
        gender = det['gender']
        emotion = det['emotion']
        facepoints = det['facepoint']
        feature = np.array(det["feature"])

        # Draw bounding box around the face
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Recognize the face based on the feature vector
        name = fr_manager.get_name(feature, thres=0.25)

        # Prepare the label text with gender, age, emotion, and confidence
        label = f'{name}, {gender}, {age}, {emotion}, Conf: {conf:.2f}'
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw the face points (landmarks) as red circles
        for (x, y) in facepoints:
            x, y = int(x), int(y)
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)  # Red circle for each face point

    return image


def main_testing_img(fd_client, fr_manager):
    # Load the image for processing
    image = cv2.imread('test.jpg')

    # Process the image
    processed_image = process_image(image, fd_client, fr_manager)

    # Save the image with bounding boxes and labels
    cv2.imwrite('detected_faces.jpg', processed_image)
    print("Saved detected faces image to 'detected_faces.jpg'")

def main_testing_video(fd_client, fr_manager):
    # Open video file (can also be a webcam feed, e.g., 0 for default webcam)
    video_input_path = 'video_testing_natalie_jagdip.mp4'
    cap = cv2.VideoCapture(video_input_path)

    # Get the frame width, height, and FPS for the output video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Set up VideoWriter to save the output video
    output_video_path = 'output_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Process each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        processed_frame = process_image(frame, fd_client, fr_manager)

        # Write the processed frame to the output video
        out.write(processed_frame)

    # Release the video capture and writer objects
    cap.release()
    out.release()

    print("Video processing complete. Saved output to 'output_video.mp4'")


if __name__ == "__main__":
    # Initialize the face detection and recognition clients
    fd_client = FaceDetectionClient(docker_url='tcp://localhost:7005')
    folder_generate_fr = "./database_fr"
    today = datetime.now()
    folder_load_fr = f"{folder_generate_fr}_npy_{today.day}_{today.month}_{today.year}/"

    # Create the face recognition database
    FRCreateDatabase(fd_client).create_database(images_folder_path=folder_generate_fr, output_prefix=".npy")

    # Load the face recognition database
    fr_manager = FaceRecognitionManagement()
    fr_manager.load_database(folder_path=folder_load_fr, output_prefix=".npy")


    # Run for image processing
    # main_testing_img(fd_client, fr_manager)

    # Run for video processing
    main_testing_video(fd_client, fr_manager)
