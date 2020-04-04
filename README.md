# Tensorflow-2-Object-Detection-API-Flask-Application
This is a flask application with tensorflow 2 object detection API deployed. The user hits the endpoint with image data and gets a json file with the format {"class": "score"} in return and the also the image with detection boxes overlay gets saved in the directory specified in the request argument

## Prerequisites and Setup
* You need to setup the [tensorflow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). Here are some of the great articles that will help you in the process, [tutorial1](https://medium.com/@marklabinski/installing-tensorflow-object-detection-api-on-windows-10-7a4eb83e1e7), [tutorial2](https://gilberttanner.com/blog/installing-the-tensorflow-object-detection-api).
* You can the download the required model from [tensorflow model zoo] and specify the saved_model path in the 'client.py' file

## Running

