from flask import Flask, request, Response
import numpy as np
import cv2
import json

from utilities import load_model, load_label_map, show_inference, parse_output_dict
from custom_np_encoder import NumpyArrayEncoder

model_path = "models/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/saved_model"
labels_path = "data/mscoco_label_map.pbtxt"

vis_threshold = 0.5
max_boxes = 20

detection_model = load_model(model_path)
category_index = load_label_map(labels_path)

# Initialize the Flask application
app = Flask(__name__)

# route http posts to this method
@app.route('/object_detection', methods=['POST'])
def infer():

    # convert image data to uint8
    nparr = np.frombuffer(request.data, np.uint8)

    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # do inderence on the image and get detections and image data with overlay
    output_dict, image_np = show_inference(detection_model, img, category_index, vis_threshold, max_boxes)
    parsed_output_dict = parse_output_dict(output_dict, category_index)

    parsed_output_dict.update({"image data": image_np})

    # add the size of the image in the response
    parsed_output_dict.update({"image size": "size={}x{}".format(img.shape[1], img.shape[0])})

    # build a response dict to send back to client
    response = parsed_output_dict
    
    # encode response
    response_encoded = json.dumps(response, cls=NumpyArrayEncoder)

    return Response(response=response_encoded, status=200, mimetype="application/json")


# start flask app
app.run(host="localhost", port=5000, debug=True)