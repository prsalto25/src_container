import cv2

# Replace this with your video source:
# For file: "video.mp4"
# For RTSP: "rtsp://username:password@ip_address:port/path"
video_source = "videos_testing/Violence_JagdipNatalie_Sembawang.mp4"

cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("❌ Failed to open video stream.")
    exit()

print("✅ Video stream opened successfully.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("📴 Stream ended or failed to grab frame.")
        break

    print("Frame shape:", frame.shape)  # (height, width, channels)

cap.release()
print("🔒 Video stream released.")
