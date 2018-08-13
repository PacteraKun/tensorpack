import pandas
from coco import COCODetection
from config import finalize_configs, config as cfg
from utils.generate_chips import Im2Chip
from tensorpack.dataflow import (imgaug, TestDataSpeed, PrefetchDataZMQ,
                                 MultiProcessMapDataZMQ, MultiThreadMapData,
                                 MapDataComponent, DataFromList, RNGDataFlow,
                                 DataFlow, ProxyDataFlow)
from utils.generate_anchors import generate_anchors
from utils.np_box_ops import area as np_area

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
    logger.info(
        "Filtered {} images which contain no non-crowd groudtruth boxes. Total #images for training: {}".
        format(num - len(imgs), len(imgs)))

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

        # augmentation:
        im, params = aug.augment_return_params(im)
        points = box_to_point8(boxes)
        points = aug.augment_coords(points, params)
        boxes = point8_to_box(points)
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
        im, boxes, klass, scale_indices, is_crowd = chip_generator.genChipMultiScale(
        )
        rets = []
        for i in range(len(im)):
            try:
                if len(boxes[i]) == 0:
                    continue
                # anchor_labels, anchor_boxes
                gt_invalid = []
                maxbox = cfg.SNIPER.VALID_RANGES[scale_indices[i]][0]
                minbox = cfg.SNIPER.VALID_RANGES[scale_indices[i]][1]
                maxbox = sys.maxsize if maxbox == -1 else maxbox
                minbox = 0 if minbox == -1 else minbox
                for box in boxes[i]:
                    w = box[2] - box[0]
                    h = box[3] - box[1]
                    if w >= maxbox or h >= maxbox or (w < minbox
                                                      and h < minbox):
                        gt_invalid.append(box)
                anchor_inputs = get_sniper_rpn_anchor_input(
                    im[i], boxes[i], is_crowd[i], gt_invalid)
                assert len(anchor_inputs) == 2

                boxes[i] = boxes[i][is_crowd[i] ==
                                    0]  # skip crowd boxes in training target
                klass[i] = klass[i][is_crowd[i] == 0]

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
            ret = [im[i]] + list(anchor_inputs) + [boxes[i], klass[i]]
            rets.append(ret)
        return rets

    
    return img

    get_sniper_train_dataflow()