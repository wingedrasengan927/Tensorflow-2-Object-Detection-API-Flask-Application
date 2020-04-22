# Tensorflow-2-Object-Detection-API-Flask-Application
This is a flask application with tensorflow 2 object detection API deployed. The user hits the endpoint with image data and gets a response which consists of detections with scores, image data with overlay, image size (can be customized).

## Prerequisites and Setup
* You need to setup the [tensorflow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). Here are some of the great articles that will help you in the process, [tutorial1](https://medium.com/@marklabinski/installing-tensorflow-object-detection-api-on-windows-10-7a4eb83e1e7), [tutorial2](https://gilberttanner.com/blog/installing-the-tensorflow-object-detection-api).
* You can the download the required model from [tensorflow model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)and specify the saved_model, labelmap path in the `main.py` file
* `client.py` is used hit the endpoint. You have to specify the url with endpoint, image_path, output_dir(the dir where you want your image with overlay to be saved. It's optional), in the `client.py` file.

## Run
* first run `python main.py`
to get the app running and then to hit the endpoint with required arguments, run `python client.py`

## Output
The response consists of 
* detections with scores
* image data with overlay. This can be decoded back to a numpy array and can be written to disk to visualize the result (provided in `client.py`)
* image size

![](https://github.com/wingedrasengan927/Tensorflow-2-Object-Detection-API-Flask-Application/blob/master/outputs/girl_image_output.jpg)

## Associated Article
https://medium.com/@ms.neerajkrishna/deploy-tensorflow-object-detection-api-on-kubernetes-with-python-flask-and-docker-7a9513dd19e4

## References
* https://github.com/tensorflow/models/tree/master/research/object_detection
* https://gist.github.com/kylehounslow/767fb72fde2ebdd010a0bf4242371594
