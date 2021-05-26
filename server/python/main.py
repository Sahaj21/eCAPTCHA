import init
import numpy as np
labels= {'angry': 0,
 'disgust': 1,
 'fear': 2,
 'happy': 3,
 'neutral': 4,
 'sad': 5,
 'surprise': 6}
labels=init.idx_to_class(labels)
import cv2
import numpy as np
from time import sleep
import torch as T
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
import urllib.request
from flask import Flask
from flask import request
import torch
import os
from flask_cors import CORS, cross_origin
import json
from flask import jsonify

app = Flask(__name__)
CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'

# add parameter map_location=torch.device('cpu')
model=model=torch.load(os.path.join(app.root_path, 'weights/weights_3.pt'))

face_classifier = cv2.CascadeClassifier(os.path.join(app.root_path,'gg.xml'))

def face_detector(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return (0,0,0,0), np.zeros((224,224), np.uint8), img
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]

    try:
        roi_gray = cv2.resize(roi_gray, (224, 224), interpolation = cv2.INTER_AREA)
    except:
        return (x,w,y,h), np.zeros((224,224), np.uint8), img
    return (x,w,y,h), roi_gray, img


validation_preprocessing = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    
])


def getimgfromurl(url):
    res = urllib.request.urlopen(url)
    im = np.asarray(bytearray(res.read()), dtype="uint8")
    im = cv2.imdecode(im, cv2.IMREAD_COLOR)
    return im



@app.route("/emotion", methods=["POST"])
def classify():

    iurl = request.json.get('imgurl')
    
    img = getimgfromurl(iurl)

    rect, face, image = face_detector(img)
    if np.sum([face]) != 0.0:
        roi=Image.fromarray(face)
        
        roi=validation_preprocessing(roi)
        
        # remove .tocuda
        preds = init.predict(roi.unsqueeze(0).to('cuda'),model)[1]
        
        label = labels[preds.item()]  

        return jsonify({'label':label,'imgurl':iurl})
    else:
        return ""



app.run(host='127.0.0.1', port=8080, debug=True)