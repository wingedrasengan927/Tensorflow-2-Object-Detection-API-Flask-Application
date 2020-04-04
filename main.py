from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2

from utilities import load_model, load_label_map, show_inference, parse_output_dict

model_path = "models/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/saved_model"
labels_path = "data/mscoco_label_map.pbtxt"

vis_threshold = 0.5
max_boxes = 20

detection_model = load_model(model_path)
category_index = load_label_map(labels_path)

# Initialize the Flask application
app = Flask(__name__)

# route http posts to this method
@app.route('/object_detection/<string:save_dir>/<string:image_name>', methods=['POST'])
def infer(save_dir, image_name):

    # convert image data to uint8
    nparr = np.frombuffer(request.data, np.uint8)

    save_path = save_dir + "/" + image_name

    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # do inderence on the image and get detections
    output_dict= show_inference(detection_model, img, save_path, category_index, vis_threshold, max_boxes)
    parsed_output_dict = parse_output_dict(output_dict, category_index)

    # add the size of the image in the response
    parsed_output_dict.update({"image size": "size={}x{}".format(img.shape[1], img.shape[0])})

    # build a response dict to send back to client
    response = parsed_output_dict
    
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


# start flask app
app.run(host="localhost", port=5000, debug=True)