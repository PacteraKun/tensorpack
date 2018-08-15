# -*- coding: utf-8 -*-
# File: box_ops.py

import tensorflow as tf
from tensorpack.tfutils.scope_utils import under_name_scope

"""
This file is modified from
https://github.com/tensorflow/models/blob/master/object_detection/core/box_list_ops.py
"""


@under_name_scope()
def area(boxes):
    """
    Args:
      boxes: nx4 floatbox

    Returns:
      n
    """
    x_min, y_min, x_max, y_max = tf.split(boxes, 4, axis=1)
    return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])


@under_name_scope()
def pairwise_intersection(boxlist1, boxlist2):
    """Compute pairwise intersection areas between boxes.

    Args:
      boxlist1: Nx4 floatbox
      boxlist2: Mx4

    Returns:
      a tensor with shape [N, M] representing pairwise intersections
    """
    x_min1, y_min1, x_max1, y_max1 = tf.split(boxlist1, 4, axis=1)
    x_min2, y_min2, x_max2, y_max2 = tf.split(boxlist2, 4, axis=1)
    all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
    all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
    all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


@under_name_scope()
def pairwise_iou(boxlist1, boxlist2):
    """Computes pairwise intersection-over-union between box collections.

    Args:
      boxlist1: Nx4 floatbox
      boxlist2: Mx4

    Returns:
      a tensor with shape [N, M] representing pairwise iou scores.
    """
    intersections = pairwise_intersection(boxlist1, boxlist2)
    areas1 = area(boxlist1)
    areas2 = area(boxlist2)
    unions = (
        tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
    return tf.where(
        tf.equal(intersections, 0.0),
        tf.zeros_like(intersections), tf.truediv(intersections, unions))

def box_voting(selected_boxes, selected_prob, pool_boxes, prob, iou_thresh=0.5):
    """Performs box voting as described in S. Gidaris and N. Komodakis, ICCV 2015.
    
    Performs box voting as described in 'Object detection via a multi-region &
    semantic segmentation-aware CNN model', Gidaris and Komodakis, ICCV 2015. For
    each box 'B' in selected_boxes, we find the set 'S' of boxes in pool_boxes
    with iou overlap >= iou_thresh. The location of B is set to the weighted
    average location of boxes in S (scores are used for weighting). And the score
    of B is set to the average score of boxes in S.
    
    Args:
        selected_boxes: These boxes are usually selected from pool_boxes using non max suppression.
        selected_prob: These prob are usually selected from prob using non max suppression.
        pool_boxes: a set of (possibly redundant) boxes before NMS.
        prob: a set of (possibly redudant) prob before NMS
        iou_thresh: (float scalar) iou threshold for matching boxes in selected_boxes and pool_boxes.
    
    Returns:
    BoxList containing averaged locations and scores for each box in selected_boxes.
    
    Raises:
    ValueError: if
        a) if iou_thresh is not in [0, 1].
    """
    if not 0.0 <= iou_thresh <= 1.0:
        raise ValueError('iou_thresh must be between 0 and 1')
    
    iou_ = pairwise_iou(selected_boxes, pool_boxes)
    print("iou_: ", iou_)
    match_indicator = tf.to_float(tf.greater(iou_, iou_thresh))
    num_matches = tf.reduce_sum(match_indicator, 1)
    print("num_matches: ", num_matches)
    # TODO(kbanoop): Handle the case where some boxes in selected_boxes do not
    # # match to any boxes in pool_boxes. For such boxes without any matches, we
    # # should return the original boxes without voting.
    if not tf.reduce_all(tf.greater(num_matches, 0)):
        return selected_boxes, selected_prob
    match_assert = tf.Assert(
        tf.reduce_all(tf.greater(num_matches, 0)),
        ['Each box in selected_boxes must match with at least one box in pool_boxes.'])
    
    scores = tf.expand_dims(prob, 1)
    scores_assert = tf.Assert(
        tf.reduce_all(tf.greater_equal(scores, 0)),
        ['Scores must be non negative.'])
    
    with tf.control_dependencies([scores_assert, match_assert]):
        sum_scores = tf.matmul(match_indicator, scores)
    averaged_scores = tf.reshape(sum_scores, [-1]) / num_matches
    box_locations = tf.matmul(match_indicator, pool_boxes * scores) / sum_scores
    
    return box_locations, averaged_scores