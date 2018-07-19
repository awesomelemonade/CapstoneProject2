'''
To-Do List:
    draw rectangles and write names on pics
    log people into database
    implement clustering
'''

import databasing as db
import Camera_to_Descriptor as camdes
import id_to_faces as idf
import clustering

import matplotlib.pyplot as plt
from camera import take_picture
import numpy as np
from dlib_models import download_model, download_predictor, load_dlib_models
#download_model()
#download_predictor()
from dlib_models import models

database = db.DescriptorDatabase('database.txt')

def log_person(name, folder_path):
    '''
    Parameters:
    name - string
    folder_path - string
        path to folder with images
    '''
    list_of_arr = camdes.file_to_descriptor(folder_path)
    descriptors = [descriptor for arr in list_of_arr for descriptor in arr]
    avg_des = db.get_avg_descriptor(descriptors)
    database.put(name, avg_des)

def labelling():
    '''
    takes pic using camera and labels faces
    '''
    pic = take_picture()
    load_dlib_models()
    list_of_arr = make_descriptor(pic[np.newaxis, :, :, :])
    descriptors = [descriptor for arr in list_of_arr for descriptor in arr]
    names = []
    for descriptor in descriptors:
        names.append(idf.id_to_faces(database, descriptor))
    #someone draw the rectangles and labels kthxbye
