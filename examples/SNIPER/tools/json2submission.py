# pedestrian-detection/tools/label/json2submission.py
import json
from pprint import pprint
import os

JSON_FILE_NAME = 'output_390000_new.json'
THRESH = 0.7
OUTPUT_FILE_NAME = 'sniper_confidence_0.7_390000_'


with open(JSON_FILE_NAME) as f:
    data = json.load(f)

data = filter(lambda x: float(x['score']) >= THRESH, data)

output_dict = [x for x in data if float(x['score']) >= THRESH]

OUTPUT_FILE_NAME += str(len(output_dict))
OUTPUT_FILE_NAME += '.txt'

with open(OUTPUT_FILE_NAME, 'w') as fw:
    for item in output_dict:
        result_string = ""
        # origin_img_id = item['image_id'] // 100
        # chip_id = item['image_id'] % 100
        # image_id = 'img%05d.jpg_%d' %(origin_img_id, chip_id) 
        # origin_img_id = item['image_id']//10000
        # scale_id= (item['image_id']%10000)//1000
        # chip_id = item['image_id']%1000
        image_id = 'img%05d.jpg' % item['image_id']
        # image_id = 'img%05d_%d_%02d.jpg'%(origin_img_id, scale_id, chip_id)
        result_string += image_id
        result_string += " "
        result_string += str(item['score'])
        for bbox in item['bbox']:
            result_string += " "
            result_string += str(bbox)
        fw.write(result_string + "\n")