#!/usr/bin/env python3
"""
cut_test.py: create cutted image and position file


"""
import os
import cv2
import numpy as np
import argparse



class Im2Chip(object):
    def __init__(self, im_file, im_path):
        self.imname = im_file
        self.impath = os.path.join(im_path, im_file)

        self.image = cv2.imread(self.impath, cv2.IMREAD_COLOR)

        self.image2x = cv2.resize(
            self.image, (0, 0), fx=2., fy=2., interpolation=cv2.INTER_LINEAR)
        self.image3x = cv2.resize(
            self.image, (0, 0), fx=3., fy=3., interpolation=cv2.INTER_LINEAR)
        self.image512 = self.__im2ChipSize(self.image)

    def genTestImg(self, length, path):
        image_slice0, image_data_0 = self.__genTestImgSingleScale(
            self.image512, length, 0)
        image_slice1, image_data_1 = self.__genTestImgSingleScale(
            self.image, length, 1)
        image_slice2, image_data_2 = self.__genTestImgSingleScale(
            self.image2x, length, 2)
        image_slice3, image_data_3 = self.__genTestImgSingleScale(
            self.image3x, length, 3)
        image_slice = {
            **image_slice0,
            **image_slice1,
            **image_slice2,
            **image_slice3
        }
        image_data = {
            **image_data_0,
            **image_data_1,
            **image_data_2,
            **image_data_3
        }
        for im_name in image_slice:
            im_path = os.path.join(path, im_name)
            # cv2.imshow('name' , image_slice[im_name])
            cv2.imwrite(im_path, image_slice[im_name])
        # with open(            os.path.join(path,self.imname), 'w') as outfile:
        #     json.dump(image_data, outfile)
        return image_data

    def __genTestImgSingleScale(self, image, length, scale):
        image_slice = {}
        image_info = {}
        [h, w] = image.shape[0:2]
        x_slice_num = int(w // length) + 1
        y_slice_num = int(h // length) + 1
        if not x_slice_num == 1:
            x_slice_num += 1
        if not y_slice_num == 1:
            y_slice_num += 1
        x_top_left_pos = np.linspace(
            0, w - length, x_slice_num, endpoint=True, dtype=int)
        y_top_left_pos = np.linspace(
            0, h - length, y_slice_num, endpoint=True, dtype=int)
        top_left_pos = [[x, y] for x in x_top_left_pos for y in y_top_left_pos]
        for i in range(len(top_left_pos)):
            slice_name = '%s_%d_%02d.jpg' % (self.imname.split('.')[0], scale,
                                             i)
            slice_data = np.array(
                image[top_left_pos[i][1]:top_left_pos[i][1] +
                      length, top_left_pos[i][0]:top_left_pos[i][0] + length])
            slice_reshape = np.zeros((length, length, 3)).astype(np.uint8)
            slice_reshape[0:slice_data.shape[0], 0:slice_data.shape[
                1], :] += slice_data
            image_slice[slice_name] = slice_reshape
            image_info[slice_name] = top_left_pos[i] + [scale]
        return image_slice, image_info

    def __im2ChipSize(self, image):
        im_max_size = max(self.image.shape[:2])
        return cv2.resize(
            image, (0, 0),
            fx=512. / im_max_size,
            fy=512. / im_max_size,
            interpolation=cv2.INTER_LINEAR)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Cut Test Files')
    parser.add_argument('--img', dest='img', help='Image folder')
    parser.add_argument(
        '--output', dest='output', help='Image output folder', default='out')
    parser.add_argument(
        '--position', dest='position', help='position file', default='position.txt')
    parser.add_argument(
        '--size', dest='size', help='crop size', default=512)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    img_folder = args.img
    output_folder = args.output
    position = args.position
    size = args.size

    img_info_total = {}

    for key in os.listdir(img_folder):
        print(key)

        image_cutter = Im2Chip(key, img_folder)
        img_info = image_cutter.genTestImg(size, output_folder)
        img_info_total.update(img_info)
    with open(position, 'w') as position_file:
        for key in img_info_total:
            position_file.write(key)
            position_file.write(' %d %d %d %d %d\n' %
                                (img_info_total[key][0], img_info_total[key][1],
                                size, size, img_info_total[key][2]))
