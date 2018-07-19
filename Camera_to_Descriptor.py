import matplotlib.pyplot as plt
from camera import take_picture
import numpy as np
from dlib_models import download_model, download_predictor, load_dlib_models
download_model()
download_predictor()
from dlib_models import models
def camera_to_descriptor():
    pic = take_picture()
    load_dlib_models()
    face_detect = models["face detect"]
    face_rec_model = models["face rec"]
    shape_predictor = models["shape predict"]
    detections = list(face_detect(pic))
    descriptor = []
    for detection in detections:
        shape = shape_predictor(pic,detection)
        descriptor.append(np.array(face_rec_model.compute_face_descriptor(pic, shape)))
    descriptor = np.array(descriptor)
    return descriptor
