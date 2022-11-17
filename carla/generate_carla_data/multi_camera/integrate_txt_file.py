import json
import os
import math


def get_txt(json_path):
    reslut = []
    for line in open(json_path, 'r'):
        reslut.append((line))
    return reslut


def get_all_txt_file(path):
    files = os.listdir(path)
    files.sort()
    return files


def get_txt_info(path):
    f = open(path)
    txt_data = json.load(f)
    return txt_data


def intergrate_txts(cam_list):

    result = []
    cam = cam_list
    for i in range(len(cam)):

        path = f'{cam[i]}/out_bbox'
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        files = get_all_txt_file(path)

        for file in files:
            file_path = path + '/' + file
            if 'gt.txt' in file_path:
                continue
            txt_data = get_txt_info(file_path)

            frame = int(file.split('.')[0])

            ID = txt_data['vehicle_id']
            bboxes = txt_data['bboxes']

            assert len(ID) == len(bboxes)

            for i in range(len(ID)):
                id = ID[i]
                left = math.floor(bboxes[i][0][0])
                top = math.floor(bboxes[i][0][1])
                width = math.floor(bboxes[i][1][0] - left)
                height = math.floor(bboxes[i][1][1] - top)
                gt = str(
                    f'{frame},{id},{left},{top},{width},{height},1,-1,-1,-1')
                result.append(gt)

        filename = f'{path}/gt.txt'
        with open(filename, mode='w') as f:
            for data in result:
                f.write(data + '\n')
