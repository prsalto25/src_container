import os
import cv2

# Local Library
from triton_client_script.yolo_ultra import YoloUltralyticsClientZMQ
from utils.stream import createFileVideoStream
from utils.stream_out import OutStream
# from utils.stream_out2 import OutStream
# from utils.stream_out_imshow import OutStream
# from utils.mysql2 import Mysql
from utils.mysql_helper import MySQLHelper
from utils.timer_gray import TimerGray

import config as p

cv2.setNumThreads(1)

# Constants
original_frame_size = (int(p.ORIGINAL_WIDTH), int(p.ORIGINAL_HEIGHT))
streamout_frame_size = (int(p.STREAMOUT_WIDTH), int(p.STREAMOUT_HEIGHT))

# Algo Mapping
class AlgosManager:
    def __init__(self, timer, cam_id, algos_dict, flag_testing=False):
        self.timer = timer
        self.algos_func = {}

        # Initialize MySQL
        self.mysql_helper = None
        if not flag_testing: 
            self.mysql_helper = MySQLHelper(
                host=p.MYSQL_HOST,
                port=p.MYSQL_PORT,
                username=p.MYSQL_USERNAME,
                password=p.MYSQL_PASSWORD,
                database=p.MYSQL_DATABASE
            )
            # Generate tickets table if not available
            tickets_field = [
                ['id', 'varchar(40)'],
                ['type', 'varchar(40)'],
                ['createdAt', 'datetime'],
                ['updatedAt', 'datetime'],
                ['assigned', 'varchar(40)'],
                ['id_account', 'varchar(40)'],
                ['id_branch', 'varchar(40)'],
                ['level', 'int(11)'],
                ['reviewed', 'varchar(45)']
            ]
            self.mysql_helper.add_table('tickets', tickets_field)

        # Initialize algorithms
        if 'fr' in algos_dict:
            from algo_helper.fr_dan1 import FR
            out_stream = OutStream(streamout_frame_size, algos_dict['fr']['stream_in'])
            self.algos_func['fr'] = FR(timer, out_stream, self.mysql_helper, algos_dict['fr'])

        if 'testing' in algos_dict: 
            from algo_helper.template_va import TemplateVA
            out_stream = OutStream(streamout_frame_size, algos_dict['testing']['stream_in'])
            self.algos_func['testing'] = TemplateVA(timer, out_stream, self.mysql_helper, algos_dict['testing'], flag_testing)

        if 'ANPR' in algos_dict:
            from algo_helper.anpr import ANPR
            key = 'ANPR'
            self.outstream = OutStream(streamout_frame_size, algos_dict[key]['stream_in'])
            # self.algos_func[key] = None
            self.algos_func[key] = ANPR(algos_dict[key], self.outstream, self.mysql_helper)

        if 'violence' in algos_dict:
            from algo_helper.violence import Violence
            key = "violence"
            print(str(algos_dict[key]["camera_name"]))
            self.outstream = OutStream(streamout_frame_size, algos_dict[key]['stream_in'])
            self.algos_func[key] = Violence(timer, self.outstream, self.mysql_helper, algos_dict[key], None)

        if 'intrusion' in algos_dict:
            key = "intrusion"
            from algo_helper.intrusion import Intrusion
            self.outstream = OutStream(streamout_frame_size, algos_dict[key]['stream_in'])
            self.algos_func[key] = Intrusion(algos_dict[key], self.outstream, self.mysql_helper)

        # hiro analytics
        if 'aod' in algos_dict:
            key = "aod"
            from algo_helper.aod import AOD
            self.outstream = OutStream(streamout_frame_size, algos_dict[key]['stream_in'])
            self.algos_func[key] = AOD(algos_dict[key], self.outstream, self.mysql_helper)

        if 'crowd' in algos_dict:
            key = "crowd"
            from algo_helper.people_count import PeopleCount
            self.outstream = OutStream(streamout_frame_size, algos_dict[key]['stream_in'])
            self.algos_func[key] = PeopleCount(algos_dict[key], self.outstream, self.mysql_helper)

        if 'vcount' in algos_dict:
            key = "vcount"
            from algo_helper.vehicle_enter_exit import VehicleEnterExit
            self.outstream = OutStream(streamout_frame_size, algos_dict[key]['stream_in'])
            self.algos_func[key] = VehicleEnterExit(algos_dict[key], self.outstream, self.mysql_helper)


    def run(self, frame, yolo_tracks, yolo_dets, original_frame):
        if 'fr' in self.algos_func:
            self.algos_func['fr'].run(frame.copy(), yolo_tracks.get('person', []))

        if 'testing' in self.algos_func:
            self.algos_func['testing'].run(frame.copy(), yolo_tracks.get('person', []))

        if 'ANPR' in self.algos_func:
            self.algos_func['ANPR'].run(frame.copy(), yolo_tracks.get('vehicle', []), original_frame.copy())

        if 'violence' in self.algos_func:
            self.algos_func['violence'].run(frame.copy(), yolo_tracks.get('person', []))

        if 'intrusion' in self.algos_func:
            self.algos_func['intrusion'].run(frame.copy(), yolo_tracks.get('person', []))

        # hiro
        if 'aod' in self.algos_func:
            # person and bag
            self.algos_func['aod'].run(frame.copy(), yolo_tracks.get('person', []), yolo_dets.get('bag', []))

        if 'crowd' in self.algos_func:
            self.algos_func['crowd'].run(frame.copy(), yolo_tracks.get('person', []))

        if 'vcount' in self.algos_func:
            self.algos_func['vcount'].run(frame.copy(), yolo_tracks.get('vehicle', []))

        

def main(cam_dict, flag_testing=False):
    print("Running: run_step3_algo.py")
    cam_id = cam_dict['camera_id']
    input_path = cam_dict['rtsp_in']
    algos_dict = cam_dict['algos']

    # Initialize video stream and components
    fvs = createFileVideoStream(input_path, max_fps=p.MAX_FPS, log_interval=p.LOG_INTERVAL, vid_skip_frame=p.VID_SKIP_FRAME, vid_restart=p.FLAG_VIDEO_LOOP)
    # fvs = createFileVideoStream(input_path, max_fps=max_fps, log_interval=5, vid_skip_frame=20, vid_restart=flag_video_loop)
    
    yolo = YoloUltralyticsClientZMQ(zmq_url=p.YOLO_ZMQ_URL, width=640, height=640)
    timer = TimerGray()
    algos_manager = AlgosManager(timer, cam_id, algos_dict, flag_testing)

    print('RUNNING VIDEO ANALYTICS.....')
    
    while True:
        timer.update()
        original_frame = fvs.read()
        if original_frame is None:
            break
        resized_frame = cv2.resize(original_frame, original_frame_size)
        # resized_frame = original_frame
        yolo_dets = yolo.detect(resized_frame)
        yolo_tracks = yolo.track(yolo_dets)
        algos_manager.run(resized_frame, yolo_tracks, yolo_dets, original_frame)



if __name__ == '__main__':
    from multiprocessing import Process
    input_dict = {
        # "testing": "/home/videos_testing/violence_nataliejagdip_thk.mp4", 
        # "crowd": "/home/src/videos_testing/Violence_JagdipNatalie_Sembawang.mp4", 
        # "aod": "/home/src/videos_testing/Left_Object_1_Cam2_1.avi",
        # "aod": "/home/src/videos_testing/Left_Object_2_Cam1_1.avi",
        # "violence": "/home/src/videos_testing/Violence_JagdipNatalie_Sembawang.mp4",
        # "intrusion": "/home/src/videos_testing/violence_nataliejagdip_thk.mp4",
        # "fr": "asdf.mp4",
        "ANPR": "/home/src/videos/2 - license plates + firearm detection - Made with Clipchamp.mp4"
    }
    flag_testing = True

    cam_dicts = []
    base_cam_id = 100
    for idx, (algo_name, video_src) in enumerate(input_dict.items()):
        cam_id = str(base_cam_id + idx)
        stream_in = f"http://localhost:8090/feed{idx+50}.ffm"
        cam_dict = {
            "rtsp_in": video_src,
            "camera_id": cam_id,
            "algos": {
                algo_name: {
                    "camera_name": cam_id,
                    "camera_id": cam_id,
                    "id_account": cam_id,
                    "id_branch": cam_id,
                    "stream_in": stream_in,
                    "attributes": [{}, {algo_name: 0}],
                    "rois": [[[0, 0], [0, 720], [1280, 720], [1280, 0]]]
                }
            }
        }
        cam_dicts.append(cam_dict)

    print("--------------------------")
    print("Cam Dicts: ", cam_dicts)
    print("--------------------------")

    processes = []
    for cam_dict in cam_dicts:
        pr = Process(target=main, args=(cam_dict, flag_testing))
        pr.start()
        processes.append(pr)

    for pr in processes:
        pr.join()

