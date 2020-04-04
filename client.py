from __future__ import print_function
import numpy as np
import requests
import json

address = 'http://localhost:5000'
url = address + '/object_detection'
img_path = 'images/girl_image.jpg'
save_dir = "outputs/"

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

def post_image(img_path, URL, save_dir):
    """ post image and return the response """
    # get image name from image path
    img_name = img_path.split("/")[-1].split(".")[0]
    save_path = save_dir + img_name + '_output.jpg'

    URL = URL + "/" + save_path
    img = open(img_path, 'rb').read()
    response = requests.post(URL, data=img, headers=headers)
    return response

def parse_response(response):
    return json.loads(response.text)

response = post_image(img_path, url, save_dir)
print(parse_response(response))