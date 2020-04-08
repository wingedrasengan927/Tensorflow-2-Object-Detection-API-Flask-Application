import numpy as np
import os
import pathlib
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pickle
import io

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
import cv2

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

def load_model(model_dir):
    # load the object detection model
    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    return model

def load_label_map(path):
    # List of the strings that is used to add correct label for each box.
    category_index = label_map_util.create_category_index_from_labelmap(path, use_display_name=True)
    return category_index

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                    for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    output_dict['detection_masks'], output_dict['detection_boxes'],
                    image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                            tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

def show_inference(model, image_np, category_index, vis_thresh, max_boxes_to_draw):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.

    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        min_score_thresh = vis_thresh,
        max_boxes_to_draw = max_boxes_to_draw,
        line_thickness=8)

    return output_dict, image_np

def parse_output_dict(output_dict, category_index):
    "parse output dict in the format {'class':'score'}"
    parsed_array = []
    parsed_dict = {}
    detection_classes = output_dict['detection_classes']
    detection_scores = output_dict['detection_scores']
    for detection_class, detection_score in zip(detection_classes, detection_scores):
        parsed_array.append({category_index[detection_class]["name"]: str(int(detection_score*100)) + "%"})
    parsed_dict.update({"detections": parsed_array})
    return parsed_dict
