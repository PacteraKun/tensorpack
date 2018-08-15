import cv2
import numpy as np
import copy
import itertools
import os
import pandas
import re
import sys

from coco import COCODetection
from config import finalize_configs, config as cfg
from utils.generate_chips import Im2Chip
from tensorpack.utils.argtools import memoized, log_once
from tensorpack.dataflow import (imgaug, TestDataSpeed, PrefetchDataZMQ,
                                 MultiProcessMapDataZMQ, MultiThreadMapData,
                                 MapDataComponent, DataFromList, RNGDataFlow,
                                 DataFlow, ProxyDataFlow)
from tensorpack.utils import logger
from utils.generate_anchors import generate_anchors
from utils.np_box_ops import area as np_area

class MalformedData(BaseException):
    pass


OUTPUT_FILE = 'train_chip_annotations.txt'
out = open(OUTPUT_FILE, 'w')

def get_sniper_train_dataflow():

    imgs = COCODetection.load_many(
        cfg.DATA.BASEDIR, cfg.DATA.TRAIN, add_gt=True, add_mask=cfg.MODE_MASK)

    # Valid training images should have at least one fg box.
    # But this filter shall not be applied for testing.
    num = len(imgs)
    imgs = list(
        filter(lambda img: len(img['boxes'][img['is_crowd'] == 0]) > 0, imgs))
    print(imgs[0])

    proposal_pickle = pandas.read_pickle(cfg.SNIPER.PRN_PRE)

    def preprocess(img):

        fname, boxes, klass, is_crowd = img['file_name'], img['boxes'], img[
            'class'], img['is_crowd']
        img_name = fname.split('/')[-1]
        print(img_name)
        img_id = int(img_name[3:-4])
        # pretrain rpn for negtive chip extraction

        proposals = proposal_pickle['boxes'][proposal_pickle['ids'].index(
            img_id)]
        proposals[2:4] += proposals[0:2]  # from [x,y,w,h] to [x1,y1,x2,y2]

        boxes = np.copy(boxes)
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        assert im is not None, fname
        im = im.astype('float32')
        # assume floatbox as input
        assert boxes.dtype == np.float32, "Loader has to return floating point boxes!"

        assert np.min(np_area(boxes)) > 0, "Some boxes have zero area!"
        chip_generator = Im2Chip(
            im,
            boxes,
            klass,
            proposals,
            cfg.SNIPER.SCALES,
            cfg.SNIPER.VALID_RANGES,
            is_crowd=is_crowd,
            chip_size=cfg.SNIPER.CHIP_SIZE,
            chip_stride=cfg.SNIPER.CHIP_STRIDE)
        im, boxes, klass, scale_indices, is_crowd = chip_generator.genChipMultiScale()
        rets = []
        for i in range(len(im)):
            try:
                if not len(boxes[i]):
                    raise MalformedData("No valid gt_boxes!")
            except MalformedData as e:
                log_once(
                    "Input {} is filtered for training: {}".format(
                        fname, str(e)), 'warn')
                ret = None
                continue

            # ret = [im[i]] + list(anchor_inputs) + [boxes[i], klass[i]
            #                                        ] + [scale_indices[i]*len(boxes[i])]
            ret = [im[i]] + [boxes[i], klass[i]]
            rets.append(ret)
        return rets
    for img in imgs:
        preprocess(img)
    
    return imgs

get_sniper_train_dataflow()