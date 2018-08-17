# SNIPER on COCO
This example provides a minimal (<2k lines) and faithful implementation of the following papers:

+ [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
+ [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)
+ [Mask R-CNN](https://arxiv.org/abs/1703.06870)
+ [An Analysis of Scale Invariance in Object Detection – SNIP](https://www.cs.umd.edu/%7Ebharat/crsnip.pdf)
+ [SNIPER: Efficient Multi-Scale Training](https://www.cs.umd.edu/%7Ebharat/sniper.pdf)

with the support of:
+ Multi-GPU / distributed training
+ [Cross-GPU BatchNorm](https://arxiv.org/abs/1711.07240)
+ [Group Normalization](https://arxiv.org/abs/1803.08494)

## Dependencies
+ Python 3; TensorFlow >= 1.6 (1.4 or 1.5 can run but may crash due to a TF bug);
+ [pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/), OpenCV.
+ Pre-trained [ImageNet ResNet model](http://models.tensorpack.com/FasterRCNN/)
  from tensorpack model zoo. Use the models with "-AlignPadding".
+ COCO data. It needs to have the following directory structure:
```
COCO/DIR/
  annotations/
    instances_train2014.json
    instances_val2014.json
    instances_minival2014.json
    instances_valminusminival2014.json
  train2014/
    COCO_train2014_*.jpg
  val2014/
    COCO_val2014_*.jpg
```
`minival` and `valminusminival` are optional. You can download them
[here](https://github.com/rbgirshick/py-faster-rcnn/blob/master/data/README.md).


## Usage
### Train:

On a single machine:
```
./train.py --config \
    MODE_MASK=False MODE_FPN=False \
    MODE_SNIPER=True \
    DATA.BASEDIR=/path/to/COCO/DIR \
    BACKBONE.WEIGHTS=/path/to/weight \
```

To run distributed training, set `TRAINER=horovod` and refer to [HorovodTrainer docs](http://tensorpack.readthedocs.io/modules/train.html#tensorpack.train.HorovodTrainer).

Options can be changed by either the command line or the `config.py` file.
Recommended configurations are listed in the table below.

The code is only valid for training with 1, 2, 4 or >=8 GPUs.
Not training with 8 GPUs may result in different performance from the table below.


To evaluate the performance of a model on COCO format data, please use the origin version of tensorpack-faster rcnn(). First you should crop testing images to your training scale using [cut_test.py](./tools/cut_test.py) and [coco_parser_test.py](./tools/coco_parser_test.py). Then test on cropped images with TEST.FRCNN_NMS_THRESH = 1, put back to origin scale and filter by using SNIPER.VALID_RANGES in config and using [ParseJson.py](./tools/ParseJson.py), parse output json file to WIDER format using [json2submission.py](./tools/json2submission.py). Finally NMS should be applied using [bbox_nms.py](./tools/bbox_nums.py).

```
./train.py --evaluate output.json --load /path/to/COCO-R50C4-MaskRCNN-Standard.npz \
--config MODE_MASK=False MODE_FPN=False \ DATA.BASEDIR=/path/to/CROPPED/IMAGE
```
Evaluation or prediction will need the same `--config` used during training.


