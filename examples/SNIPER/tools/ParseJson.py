#!/usr/bin/env python3
"""
ParseJson: Parse Json of cropped image back into origin image
only support 1x 2x 3x and to 512.0 scale
change init if want to change crop size
need to provide position file


"""
import json
from pprint import pprint
import cv2
# position file
POSITION = 'val_partial_position.txt'
position_dict = {}
with open(POSITION, 'r') as rf:
    rf_lines = rf.readlines()
    for rf_line in rf_lines:
        rf_split = rf_line.split()
        position_dict[rf_split[0]] = list(map(int, rf_split[1:5]))
# scale dict to filter results
# based on sniper config
range_dict = [(300, 10000), (128, 320), (80, 160), (0, 100)]
scale_dict = [0, 1, 2, 3]
# input json file
with open('output_390000_new.json') as f:
    datas = json.load(f)
output = []
for data in datas:
    print(data['image_id'])
    chip_id = data['image_id'] % 1000
    scale_id = (data['image_id'] // 1000) % 10
    image_id = data['image_id'] // 10000
    position_id = 'img%05d_%d_%02d.jpg' % (image_id, scale_id, chip_id)
    if scale_id == 0:
        image_max_len = max(cv2.imread('val/img%05d.jpg' % image_id).shape)
        scale_dict[0] = 512 / image_max_len
        # print(scale_dict)
    w = data['bbox'][2] / scale_dict[scale_id]
    h = data['bbox'][3] / scale_dict[scale_id]
    max_wh = max(w, h)
    minlen = range_dict[scale_id][0]
    maxlen = range_dict[scale_id][1]
    if max_wh > minlen and max_wh < maxlen:
        p_w = position_dict[position_id][0]
        p_h = position_dict[position_id][1]
        data['image_id'] = image_id
        data['bbox'][0] += p_w
        data['bbox'][1] += p_h
        for i in range(len(data['bbox'])):
            data['bbox'][i] /= scale_dict[scale_id]
        output.append(data)
# output json
with open("filtered_390000.json", "w") as write_file:
    json.dump(output, write_file)
