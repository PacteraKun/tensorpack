#!/usr/bin/env python3
"""
generate_chips.py: prepare chip for sniper
can run independently but need to provide factory function for data

"""
__author__ = "Jiaqi Cai"
__email__ = "jiaqi.cai22@pactera.com"

import cv2
import numpy as np
import json


class Im2Chip(object):
    def __init__(self,
                 image,
                 gt_list,
                 gt_label,
                 rp_list,
                 scale_list,
                 box_range_list,
                 is_crowd,
                 chip_size=512,
                 chip_stride=32):
        """
        take origin image and ground truth, pre trained RPN proposal,
        target scale and chip configure as input
        """
        assert len(scale_list) == len(
            box_range_list), "scale number does not match valid range"
        self.image = image
        self.gt_list = gt_list
        self.gt_label = gt_label
        self.rp_list = rp_list
        self.scale_list = scale_list
        self.box_range_list = box_range_list
        self.chip_size = chip_size
        self.chip_stride = chip_stride
        self.is_crowd = is_crowd

        # self.chip_candidates = self.__genChipCandidate(self.image.shape)

    def genChipMultiScale(self):
        """
        combine selected chips on different scales

        return:
        chips_all : all selected chips
        chips_gt_all : ground truth on selected chips
        chips_gt_label_all : ground truth class label 
        scale_index_all : scale index(in config file) of chip
        is_crowd_all : iscrowd label of ground truth
        """
        h, w, channel = self.image.shape
        scale_list = self.scale_list.copy()
        scale_list[-1] /= max(w, h)
        chips_all = []
        chips_gt_all = []
        chips_gt_label_all = []
        scale_index_all = []
        is_crowd_all = []
        for i in range(len(scale_list)):
            chips, chips_gt, chips_gt_label, is_crowd = self.__genChip(
                scale_list[i], self.box_range_list[i])
            scale_indexes = [i] * len(chips)
            chips_all += chips
            chips_gt_all += chips_gt
            chips_gt_label_all += chips_gt_label
            scale_index_all += scale_indexes
            is_crowd_all += is_crowd
            # break
        # for i in range(len(chips_all)):
        #     for j in range(len(chips_gt_all[i])):
        #         print(chips_gt_all[i])
        #         cv2.rectangle(chips_all[i],
        #                       (int(chips_gt_all[i][j][0]), int(chips_gt_all[i][j][1])),
        #                       (int(chips_gt_all[i][j][2]), int(chips_gt_all[i][j][3])),
        #                       (255, 255, 0), 1)
        #     cv2.imshow('name', chips_all[i])
        #     cv2.waitKey(0)
        return chips_all, chips_gt_all, chips_gt_label_all, scale_index_all, is_crowd_all

    def __genChipCandidate(self, shape):
        """
        generate chips on different certain shape
        input:
        shape : shape of image
        
        return:
        generated chips
        """
        s = self.chip_stride
        # cv2 have revised order of shape
        h = shape[0]
        w = shape[1]
        x_inds = np.arange(0, max(w - self.chip_size + s, s), s, dtype=int)
        xlen = len(x_inds)
        y_inds = np.arange(0, max(h - self.chip_size + s, s), s, dtype=int)
        ylen = len(y_inds)
        x_inds = np.array([
            x_inds,
        ] * ylen).flatten()
        y_inds = np.array([
            y_inds,
        ] * xlen).flatten(order='F')
        w_inds = np.ones(len(x_inds), dtype=int) * self.chip_size
        h_inds = np.ones(len(y_inds), dtype=int) * self.chip_size
        x2_inds = w_inds + x_inds
        y2_inds = h_inds + y_inds
        chips = np.vstack((x_inds, y_inds, x2_inds, y2_inds)).transpose()
        for chip in chips:
            if chip[2] > w:
                chip[2] = w
            if chip[3] > h:
                chip[3] = h
        np.random.shuffle(chips)
        return chips

    def __im2ChipSize(self, image):
        """
        rescale image to chip size
        input : image to rescale
        return : rescaled image
        """
        im_max_size = max(self.image.shape[:2])
        return cv2.resize(
            image, (0, 0),
            fx=512. / im_max_size,
            fy=512. / im_max_size,
            interpolation=cv2.INTER_LINEAR)

    def __contain_single_box(self, chip, box):
        """
        test if box in certain chip
        """
        # print(chip, box)
        if chip[0] <= box[0] and chip[1] <= box[1] and chip[2] >= box[2] and chip[3] >= box[3]:
            return True
        else:
            return False

    def __overlap(self, chip_candidates, gt_list):
        """
        calculate if ground truth in chip
        input :
        gt_list : a list of ground truths
        chip_candidates : a list of chips
        output : 
        candidate_contains : ground truth index list in chips
        candidate_contains_size : number of ground truths in chips
        gt2candidates : revised list from ground truth to chip
        """
        gt2candidates = {}
        candidate_contains_size = []
        candidate_contains = []
        for i in range(len(chip_candidates)):
            contain = set()
            for j in range(len(gt_list)):
                if self.__contain_single_box(chip_candidates[i], gt_list[j]):
                    contain.add(j)
                    if j in gt2candidates:
                        gt2candidates[j].add(i)
                    else:
                        gt2candidates[j] = set()
                        gt2candidates[j].add(i)
            candidate_contains.append(contain)
            candidate_contains_size.append(len(contain))
        return candidate_contains, candidate_contains_size, gt2candidates

    def __genChip(self, scale, s_range):
        """
        generate chip on certain scale
        input :
        scale : scale of chip
        s_range : valid ground truth range

        return :
        chips : selected chips
        chips_gts : ground truth on selected chips
        chips_gts_label : ground truth label on selected chips
        is_crowd : ground truth iscrowd label
        """
        image_scaled = cv2.resize(self.image, (0, 0), fx=scale, fy=scale)
        chip_candidates_scaled = self.__genChipCandidate(image_scaled.shape)
        box_min = 0 if s_range[0] == -1 else s_range[0]
        box_max = max(image_scaled.shape) if s_range[1] == -1 else s_range[1]
        gt_filtered_index = np.argwhere(
            np.logical_and.reduce(
                (np.logical_or(
                    self.gt_list[:, 2] - self.gt_list[:, 0] >= box_min,
                    self.gt_list[:, 3] - self.gt_list[:, 1] >= box_min),
                 self.gt_list[:, 2] - self.gt_list[:, 0] < box_max,
                 self.gt_list[:, 3] - self.gt_list[:, 1] <
                 box_max))).flatten().tolist()
        gt_filtered = self.gt_list[gt_filtered_index]
        gt_filtered_scaled = gt_filtered * scale
        chips_pos = self.__genPosChips(chip_candidates_scaled,
                                       gt_filtered_scaled)
        # print(chips_pos)
        rp_filtered = np.array([
            s for s in self.rp_list
            if ((s[2] - s[0]) >= box_min or (s[3] - s[1]) >= box_min) and
            (s[2] - s[0]) < box_max and (s[3] - s[1]) < box_max
        ])
        rp_filtered *= scale
        chips_neg = self.__genNegChips(chip_candidates_scaled, chips_pos,
                                       rp_filtered, 40, 2)
        # chips_shape = chip_candidates_scaled[chips_neg + chips_pos].astype(int)
        chips, chips_gts, chips_gts_label, is_crowd = self.__genChipsGt(
            chip_candidates_scaled[chips_pos + chips_neg], image_scaled, scale)

        return chips, chips_gts, chips_gts_label, is_crowd

    def __genPosChips(self, chip_candidates, gt_filtered):
        """
        select positive chips
        
        input : 
        chip_candidates : chips
        gt_filtered : valid ground truths

        return: 
        chips : selected chips 
        """
        gt_boxes = gt_filtered
        candidate_contains, candidate_contains_size, gt2candidates = self.__overlap(
            chip_candidates, gt_boxes)
        # print(gt2candidates)
        chips = []
        gt_checked = set()
        candidate_contains_max = np.argmax(candidate_contains_size)
        while not (candidate_contains_size[candidate_contains_max] == 0):
            chips.append(candidate_contains_max)
            # chip_gt = []
            gt_inside_list = list(candidate_contains[candidate_contains_max])
            for gt_inside_index in gt_inside_list:
                # delete from candidate contain list
                if gt_inside_index not in gt_checked:
                    for candidate_index in gt2candidates[gt_inside_index]:
                        candidate_contains_size[candidate_index] -= 1
                    gt_checked.add(gt_inside_index)
            candidate_contains_max = np.argmax(candidate_contains_size)
        return chips

    def __genNegChips(self, chip_candidates, chips_pos, rp_filtered, rpn_count,
                      n):
        """
        select negative chips
        
        input : 
        chip_candidates : chips
        chips_pos : selected positive chip
        rp_filtered : filtered proposal from pretrained RPN
        rpn_count : least number of proposals in selected neg chips
        n : number of negative chips selected

        return : selected negative chips 
        """
        candidate_contains, candidate_contains_size, rp2candidates = self.__overlap(
            chip_candidates, rp_filtered)
        checked_rp = set()
        for chosen_chip in chips_pos:
            for rp in candidate_contains[chosen_chip]:
                if rp not in checked_rp:
                    checked_rp.add(rp)
                    for candidate in rp2candidates[rp]:
                        candidate_contains_size[candidate] -= 1
        candidate_contains_size = np.array(candidate_contains_size)
        chip_neg = np.argwhere(
            candidate_contains_size >= rpn_count).flatten().tolist()
        np.random.shuffle(chip_neg)
        # print(chip_neg)
        return chip_neg[0:n]

    def __genChipsGt(self, chips_shape, image, scale):
        """
        generate new cropped ground truth on selected chips
        input :
        chips_shape : selected chips pos
        image : scaled image
        scale : scale of chip

        output :
        chips : selected chip data
        chip_gts : cropped ground truth on selected chip
        chip_gt_labels : ground truth label
        is_crowd_all : ground truth iscrowd label
        """
        gt_boxes = self.gt_list * scale
        chips = [np.array(image[s[1]:s[3], s[0]:s[2]]) for s in chips_shape]
        chip_gts = []
        chip_gt_labels = []
        is_crowd_all = []
        for i in range(len(chips_shape)):
            chip = chips_shape[i]
            chip_gt = []
            chip_gt_label = []
            is_crowd = []
            for j in range(len(gt_boxes)):
                intersection = self.__intersection(gt_boxes[j], chip)
                if intersection is not None:
                    intersection[0:2] -= chip[0:2]
                    intersection[2:4] -= chip[0:2]
                    chip_gt.append(intersection)
                    chip_gt_label.append(self.gt_label[j])
                    is_crowd.append(self.is_crowd[j])
            chip_gts.append(np.array(chip_gt))
            chip_gt_labels.append(np.array(chip_gt_label))
            is_crowd_all.append(np.array(is_crowd))
        return chips, chip_gts, chip_gt_labels, is_crowd_all

    # counting intersection for two boxes
    # each box is a N*4 array/tuple
    # order is [x1, y1, x2, y2]
    def __intersection(self, A, B):
        """
        count intersection area of two boxes
        input :
        A, B : two x1, y1, x2, y2 boxes
        return : overlap box
        """
        top_left = np.array([max(A[0], B[0]), max(A[1], B[1])])
        bottom_right = np.array([min(A[2], B[2]), min(A[3], B[3])])
        if np.all(top_left < bottom_right):
            return [*top_left, *bottom_right]
        else:
            return None


if __name__ == '__main__':
    import os
    import json
    JSON_FILE = '../../../resource/coco/annotations/instances_train2014.json'
    IMG_FOLDER = '../../../resource/coco/coco_train2014/'
    IMG_NAME = 'img08456.jpg'
    img_path = os.path.join(IMG_FOLDER, IMG_NAME)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # img_resized = cv2.resize(img, (512, 513))
    box_range_list = [[0, 100000], [0, 100000], [0, 100000]]
    train_boxes = np.array(
        [[362, 414, 99, 205], [441, 391, 94, 227], [369, 294, 54, 127],
         [831, 257, 49, 136], [1107, 268, 62, 148], [1165, 266, 49, 149], [
             1268, 245, 65, 165
         ], [1222, 200, 43, 129], [746, 185, 40, 118], [807, 187, 32, 110],
         [831, 193, 36, 109], [527, 163, 33, 85], [965, 251, 64, 168]])
    train_boxes[:, 2:4] += train_boxes[:, 0:2]
    train_labels = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
    is_crowd = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    cutter = Im2Chip(img, train_boxes, train_labels, [], [1, 2, 512.0],
                     box_range_list, is_crowd, 512, 32)
    cutter.genChipMultiScale()
