import matplotlib.pyplot as plt
from camera import take_picture
import numpy as np
import os
from dlib_models import download_model, download_predictor, load_dlib_models
download_model()
download_predictor()
load_dlib_models()
from dlib_models import models
import cv2
from skimage import io

def camera_to_descriptor():
    pic = take_picture()
    load_dlib_models()
    return make_descriptor(pic[np.newaxis, :, :, :])

def make_descriptor(list_of_arr):
    main_descriptor = []
    for arr in list_of_arr:
        face_detect = models["face detect"]
        face_rec_model = models["face rec"]
        shape_predictor = models["shape predict"]
        descriptor = []
        detections = list(face_detect(arr))
        for detection in detections:
            shape = shape_predictor(arr,detection)
            descriptor.append(np.array(face_rec_model.compute_face_descriptor(arr, shape)))
        main_descriptor.append(np.array(descriptor))
    return main_descriptor, detections

def file_to_descriptor(folder):
    list_of_arr = []
    for pic in os.listdir(folder):
        if pic == folder + '/.DS_Store':
            continue
        picture = io.imread(folder + '/' + pic)
        list_of_arr.append(np.array(picture))
    return make_descriptor(list_of_arr)
