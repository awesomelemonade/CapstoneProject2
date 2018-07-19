import matplotlib.pyplot as plt
from camera import take_picture
import numpy as np
from dlib_models import download_model, download_predictor, load_dlib_models
download_model()
download_predictor()
from dlib_models import models
from PIL import Image


def camera_to_descriptor():
    pic = take_picture()
    load_dlib_models()
    return make_descriptor(pic[np.newaxis, :, :, :])

def make_descriptor(list_of_arr):
    for arr in list_of_arr:
        face_detect = models["face detect"]
        face_rec_model = models["face rec"]
        shape_predictor = models["shape predict"]
        detections = list(face_detect(arr))
        descriptor = []
        for detection in detections:
            shape = shape_predictor(arr,detection)
            descriptor.append(np.array(face_rec_model.compute_face_descriptor(arr, shape)))
        descriptor = np.array(descriptor)
        return descriptor

def file_to_descriptor():
    folder = "../Pictures"
    list_of_arr = []
    for pic in folder:
        picture = Image.open(pic)
        list_of_arr.append(np.array(pic))
    return make_descriptor(list_of_arr)
