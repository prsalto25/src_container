import cv2
# cv2.setNumThreads(1)

class OutStream:
    def __init__(self, out_xy, output_path, cam_name="default"):
        self.out_xy = out_xy
        self.cam_name = cam_name
        cv2.namedWindow(self.cam_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.cam_name, self.out_xy[0], self.out_xy[1])

    def write(self, frame):
        resized_frame = cv2.resize(frame, self.out_xy)
        cv2.imshow(self.cam_name, resized_frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Display window closed with 'q' key.")
