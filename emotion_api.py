# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py
from __future__ import division, print_function
import io


# coding=utf-8
import sys
import os
import glob
import re
import scipy
import io
from PIL import Image
import base64
import tensorflow as tf
import align.detect_face

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.models import model_from_json

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import numpy as np
import cv2
from keras.preprocessing import image
import time
import datetime

#firebase
import pyrebase
config = {
  "apiKey": "AIzaSyBddsF7yFPuzuyhUviRCn4jRcHtqNDowWY",
  "authDomain": "matar-184107.firebaseapp.com",
  "databaseURL": "https://matar-184107.firebaseio.com",
  "storageBucket": "matar-184107.appspot.com",
  "serviceAccount": "matar-184107-firebase-adminsdk-5yogu-f971d63d12.json"
}
firebase = pyrebase.initialize_app(config)

db = firebase.database()

# initialize our Flask application and the Keras model
emotion_api = Flask(__name__)

# facenet
image_size=160
margin= 44
gpu_memory_fraction=1.0

def load_and_align_data(img, image_size,margin, gpu_memory_fraction):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    #img = scipy.misc.imread(os.path.expanduser(image_path))
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if (len(bounding_boxes)==0):
        bb=0
        have_face = 0
    else:
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2 - bb[0], img_size[1])
        bb[3] = np.minimum(det[3]+margin/2 - bb[1], img_size[0])
        have_face = 1
    return bb,have_face


model = model_from_json(open("models/model_4layer_2_2_pool.json", "r").read())
model.load_weights('models/model_4layer_2_2_pool.h5') #load weights


def model_predict(img, model):
#facenet
    #img = cv2.imread(img_path)
    detect_face, have_face= load_and_align_data(img,image_size,margin,gpu_memory_fraction)
    preds = []
    detect = []
    if (have_face!=0):
        detect_face = np.reshape(detect_face,(-1,4))

        for (x,y,w,h) in detect_face:
            detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
            detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48

            img_pixels = image.img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels, axis = 0)

            img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

            # store probabilities of 7 expressions
            predictions = model.predict(img_pixels)
            preds.append(predictions)
            detect.append(detect_face)
    return preds,have_face, detect

def decode_base64(data):
    """Decode base64, padding being optional.

    :param data: Base64 data as an ASCII byte string
    :returns: The decoded byte string.

    """
    missing_padding = len(data) % 4
    if missing_padding != 0:
        data += b'='* (4 - missing_padding)
    return base64.b64decode(data)

@emotion_api.route('/predict', methods = ['POST'])
def predict():
    data = {"success": False}
    if request.method == "POST":
        # Get the file from post request
        f = request.form["image"]
        classId = request.form["classId"]
        if f == '':
            data = {"status": '400', "message": 'Missing image form'}
            return jsonify(data)
        if classId == '':
            data = {"status": '400', "message": 'Missing classID form'}
            return jsonify(data)
        data = {"classID: ": classId}
        f = f.replace("\\n","\n")
        f = str.encode(f)
        missing_padding = len(f) % 4
        if missing_padding == 0:
            try:
                img=base64.b64decode (f)
                img = Image.open(io.BytesIO(img))
                img = np.array(img)
        # Make prediction
                emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
                # data["position"] = []
                # data["result"] = []
                data["predictions"] = []
                preds, have_face, detect_face = model_predict(img, model)


                if (have_face != 0):
                    for j in range(len(preds)):
                        # data["position"] = [{"x": float(detect_face[j][0][0]), "y": float(detect_face[j][0][1]),
                        #                      "w": float(detect_face[j][0][2]), "h": float(detect_face[j][0][3])}]
                        max_index = np.argmax(preds[j][0])
                        data["result"] = [{"result": emotions[max_index]}]

                        for i in range(len(emotions)):
                            r = {"label": emotions[i], "probability": round(preds[j][0][i] * 100, 10)}
                            data["predictions"].append(r)

                data["success"] = True
                ts = time.time()
                timenow = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                data["time_created"] = timenow
                db.child("emotion_data").child(classId).push(data)
                # db.child("realtime_data").child(classId).update(data)
            # except TypeError:
        #    data = {"status": '400'}
            except OSError:
                data = {"status": '400', "message": 'Something wrong with your base64'}
        else:
            data = {"status": '400', "message": 'Wrong Image Padding'}
    return jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."))
	#load_model()
	http_server = WSGIServer(('0.0.0.0', 5100), emotion_api)
	http_server.serve_forever()
