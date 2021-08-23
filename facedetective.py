from create_vector import Model_drowsi
from flask.scaffold import F
from werkzeug.utils import secure_filename
import numpy as np
from flask import Flask, render_template, Response, request, redirect
import cv2 
from drowsiness import *
from drowsiness_detection import *
from create_vector import *



app = Flask(__name__)

v=Video()
v_2=Video_2()
v_model=Model_drowsi()


@app.route('/')
def index():
    return render_template('index.html')

#webcam
@app.route('/camera/<drowsi>')
def camera(drowsi):
    if(drowsi=="1"):
        v_2.loop_2=True
        return Response(v_2.detect(0,0),mimetype='multipart/x-mixed-replace; boundary=frame')
    elif drowsi=="0":
        v.loop=True
        return Response(v.drowsi(0,0),mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        v_model.loop=True
        return Response(v_model.drowsi_model(0,0),mimetype='multipart/x-mixed-replace; boundary=frame')

#video
@app.route('/video/<nome_video>/<drowsi>')
def video(nome_video,drowsi):
    if(drowsi=="1"):
        v_2.loop_2=True
        return Response(v_2.detect(1,nome_video),mimetype='multipart/x-mixed-replace; boundary=frame')
    elif drowsi=="0":
        v.loop=True
        return Response(v.drowsi(1,nome_video),mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        v_model.loop=True
        return Response(v_model.drowsi_model(1,nome_video),mimetype='multipart/x-mixed-replace; boundary=frame')

#pause video or camera
@app.route('/pausar')
def pausar():
    v.loop=False
    v_2.loop_2=False
    v_model.loop=False
    return ""
if __name__=="__main__":
    app.run(debug=True)





