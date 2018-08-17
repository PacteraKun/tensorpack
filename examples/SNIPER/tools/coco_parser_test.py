#!/usr/bin/env python3
"""
coco_parser_test.py : create coco json for images

This is a tool for creating COCO json file for unlabeled images.
No annotation file required
target image size : 512 x 512
"""
__author__ = "Jiaqi Cai"
__email__ = "jiaqi.cai22@pactera.com"

import os
import cv2
import argparse
import json
import os
import cv2

PATH = 'cutted'
# Modify to image path

json_out = {
    "info": {
        "description": "COCO 2014 Dataset",
        "url": "http://cocodataset.org",
        "version": "1.0",
        "year": 2014,
        "contributor": "COCO Consortium",
        "date_created": "2017/09/01"
    },
    "images": [],
    "licenses": [],
    "categories": [{
        "supercategory": "none",
        "id": 1,
        "name": "person"
    }, {
        "supercategory": "none",
        "id": 2,
        "name": "cyclist"
    }]
}

for image_name in os.listdir(PATH):
    print(image_name)
    # only valid for image with name 'img[1-9]{5}_[1-9]{1}_[1-9]{2}.jpg'
    # need to be modified when being changed to another name
    # change the id rule when go to another kind of image
    _id = image_name.split('.')[0][3:].split('_')
    id_image =int('%s%s%03d' %(_id[0], _id[1], int(_id[2])))

    # suppose the height and width is fixed
    # if not, use cv2/PIL to read image and get image size    
    image_json = {
        "license": 6,
        "file_name": image_name,
        "height": 512,
        "width": 512,
        "id": id_image
    }
    json_out['images'].append(image_json)

with open('data_train.json', 'w') as outfile:
    json.dump(json_out, outfile)
                                     