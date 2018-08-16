import numpy as np
from imutils.object_detection import non_max_suppression

THRESH = 0.7
SUBMISSION_FILE = 'sniper_confidence_0.4_390000_filtered_5906.txt'
OUTPUT_FILE = 'submission_ensemble_390000_filtered_' + str(THRESH) + ".txt"


def non_max_suppression_fast(boxes, probs=None, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick]


def get_coordinate(x, y, w, h):
    return x, y, x + w, y + h


def read_bbox(submission_file):
    all_bbox = dict()
    with open(submission_file, 'r', encoding='utf-8') as fr:
        for idx, line in enumerate(fr):
            parts = line.split(" ")
            x1, y1, x2, y2 = get_coordinate(float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]))
            if parts[0] in all_bbox.keys():
                all_bbox[parts[0]].append((x1, y1, x2, y2, float(parts[1])))
            else:
                all_bbox[parts[0]] = list()
                all_bbox[parts[0]].append((x1, y1, x2, y2, float(parts[1])))

    fr.close()
    return all_bbox


def bbox_to_images(bbox):
    images = list()
    for key, value in bbox.items():
        one_image = (key, np.asarray(value))
        images.append(one_image)

    return images


# loop over the images
all_box = read_bbox(SUBMISSION_FILE)
"""
'img06122.jpg': 
[(1543.76794434, 294.937072754, 1609.1029052775, 379.47564697279995, 0.9614292), 
(1547.41760254, 297.940643311, 1602.6791992197, 363.4836425786, 0.48836288), 
(483.551971436, 221.907653809, 516.4211425786, 290.0017395024, 0.42841634), 
(697.461730957, 208.688247681, 722.3622436523, 288.6661682132, 0.80467176), 
(902.461608887, 201.461975098, 940.9091796878, 248.1532287601, 0.42420116), 
(484.223114014, 235.124008179, 510.01104736359997, 299.5354919436, 0.81936955), 
(908.549377441, 195.727310181, 932.9542236324, 241.93359375030002, 0.87299836), 
(872.921203613, 200.903030396, 891.8666381833, 252.541702271, 0.85704684), 
(926.151489258, 196.695953369, 945.2242431642, 241.85830688459998, 0.41192102), 
(910.424377441, 207.047485352, 928.8771362301001, 249.7248382573, 0.4464341), 
(848.827270508, 205.648757935, 865.6982421877, 250.6044769291, 0.7687473)]
"""
img = bbox_to_images(all_box)

with open(OUTPUT_FILE, "w") as fw:
    count = 0
    for (imagePath, boundingBoxes) in img:
        # load the image and clone it
        print("[x] %d initial bounding boxes" % (len(boundingBoxes)))
        np_bounding_box = np.asarray(boundingBoxes)
        # print(np_bounding_box[:, -1])
        # perform non-maximum suppression on the bounding boxes
        pick = non_max_suppression_fast(boundingBoxes, np_bounding_box[:, -1], THRESH)
        print("[x] after applying non-maximum, %d bounding boxes" % (len(pick)))
        # print(imagePath)
        # print(pick)
        for bbox in pick:
            result_str = imagePath + " " + str(bbox[4]) + " " + str(bbox[0]) + " " + str(bbox[1]) + " " + str(bbox[2] - bbox[0]) + " " + str(bbox[3] - bbox[1]) + "\n"
            # print(result_str)
            count += 1
            fw.write(result_str)

    print("Total BBOX Number: " + str(count))
fw.close()
