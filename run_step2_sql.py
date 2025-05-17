import multiprocessing as mp
import time
import json
import ast

# from utils.mysql2 import Mysql
from utils.mysql_helper import MySQLHelper
from run_step3_algo import main

import config as p


ALGO_DICT = {
    # 0: "fr",
    # 2: "loiter",
    # 4: "parking",
    # 5: "speed",
    # 8: "wrongway",
    12: "crowd",
    # 13: "anpr",
    # 16: "aod",
    # 17: "intrusion",
    # 18: "attendance",
    # 19: "violence",
    # 22: "queue",
    # 26: "vcount",
    # 28: "Animals",
    # 29: "crash",
    # 32: "Clothing",
    # 35: "weapon",
    # 38: "Collapse",
    # 39: "fire",
    # 43: "CrowdCount",
    # 48: "purseSnatching",
    # 51: "personTrack",
    # 59: "EntryExit",
    # 66: "firesmoke",
    # 70: "Congestion",
    # 71: "ppe",
    # 82: "suicideTendency",
    # 84: "DrinkersGroupIdetification",
    # 88: "zigzagDriving",
    # 92: "Jaywalk",
    # 111: "pf_edge_crossing",
    # 112: "jump_on_track",
    # 115: "pedestrain",
    # 117: "Antiintrusion",
    # 118: "bag",
    # 119: "climbing",
    # 120: "electronic",
}

mysql = MySQLHelper(
    host=p.MYSQL_HOST,
    port=p.MYSQL_PORT,
    username=p.MYSQL_USERNAME,
    password=p.MYSQL_PASSWORD,
    database=p.MYSQL_DATABASE
)

def get_rtsp_dict():
    def edit_path(rtsp):
        if rtsp[:4] in {'rtsp', 'http'}:
            return rtsp
        else:
            #return '/home/videos/' + rtsp[27:]
            return rtsp.replace('/usr/src/app/resources/', '/home/videos/').replace('/home/nodejs/app/resources/', '/home/videos/')

    cmd = 'select id, name, rtsp_in, pic_width, pic_height from cameras'
    things = mysql.run_fetch(cmd)
    rtsp_dict = {}
    for camera_id, name, rtsp_in, pic_width, pic_height in things:
        rtsp_dict[camera_id] = [name, edit_path(rtsp_in), (pic_width, pic_height)]
    return rtsp_dict

def reset_stream_url(num, id_, camera_id, algo_id):
    mysql.run(f'update relations set http_out="http://{p.HTTP_OUT_IP}:{p.HTTP_OUT_PORT}/stream{num}.mjpeg" where id="{id_}" and camera_id="{camera_id}" and algo_id="{algo_id}"')

def convert_roi(roi_id, cam_size):
    if roi_id is not None:
        rois = [[[int(i['x']/cam_size[0]*p.ORIGINAL_WIDTH), int(i['y']/cam_size[1]*p.ORIGINAL_HEIGHT)] for i in ast.literal_eval(str(roi_id))]]
    else:
        rois = [[[0,0],[p.ORIGINAL_WIDTH,0],[p.ORIGINAL_WIDTH,p.ORIGINAL_HEIGHT],[0,p.ORIGINAL_HEIGHT]]]
    return rois

def get_cam_dict(rtsp_dict):
    stream_num = 0
    cmd = 'select id, camera_id, algo_id, roi_id, atributes, id_account, id_branch, stream, createdAt, updatedAt, http_out from relations'
    things = mysql.run_fetch(cmd)
    cam_dict = {}
    for id_, camera_id, algo_id, roi_id, atributes, id_account, id_branch, stream, createdAt, updatedAt, http_out in things:
        if algo_id not in ALGO_DICT:
            print(f"algo_id {algo_id} is not in ALGO_DICT {ALGO_DICT}")
            continue
        algo_name = ALGO_DICT[algo_id]
        reset_stream_url(stream_num, id_, camera_id, algo_id)
        if camera_id not in cam_dict:
            cam_dict[camera_id] = {}
            cam_dict[camera_id]['camera_id'] = camera_id
            cam_dict[camera_id]['rtsp_in'] = rtsp_dict[camera_id][1]
            cam_dict[camera_id]['algos'] = {}
        cam_dict[camera_id]['algos'][algo_name] = {}
        cam_dict[camera_id]['algos'][algo_name]['rois'] = convert_roi(roi_id, rtsp_dict[camera_id][2])
        cam_dict[camera_id]['algos'][algo_name]['atributes'] = json.loads(atributes)
        cam_dict[camera_id]['algos'][algo_name]['algo_name'] = algo_name
        cam_dict[camera_id]['algos'][algo_name]['algo_id'] = algo_id
        cam_dict[camera_id]['algos'][algo_name]['camera_id'] = camera_id
        cam_dict[camera_id]['algos'][algo_name]['camera_name'] = rtsp_dict[camera_id][0]
        cam_dict[camera_id]['algos'][algo_name]['id_account'] = id_account
        cam_dict[camera_id]['algos'][algo_name]['id_branch'] = id_branch
        cam_dict[camera_id]['algos'][algo_name]['stream_in'] = f"http://{p.STREAM_IP}:{p.STREAM_PORT}/feed{stream_num}.ffm"
        stream_num += 1
    return cam_dict


if __name__ == '__main__':
    print("Running: run_step2_sql.py")
    rtsp_dict = get_rtsp_dict()
    cam_dict = get_cam_dict(rtsp_dict)
    mysql.close()

    print(f'rtsp_dict : {rtsp_dict}')
    print(f'cam_dict : {cam_dict}')
    print('--algos setting--')

    for camera_id, values in cam_dict.items():
        print(f'starting camera_id {camera_id} ...\n')
        p = mp.Process(target=main, args=(values,))
        p.daemon = True
        p.start()


    while True:
        time.sleep(9999)
    
