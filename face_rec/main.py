'''
To-Do List:
    draw rectangles and write names on pics
    log people into database
    implement clustering
'''

import face_rec.databasing as db
import face_rec.Camera_to_Descriptor as camdes
import face_rec.id_to_faces as idf
#import face_rec.clustering
import cv2

import matplotlib.pyplot as plt
from camera import take_picture
import numpy as np
from dlib_models import download_model, download_predictor, load_dlib_models
#download_model()
#download_predictor()
from dlib_models import models
from matplotlib.patches import Rectangle

database = db.DescriptorDatabase('database.txt')
database.load()

def log_person(name, folder_path):
    '''
    Parameters:
    name - string
    folder_path - string
        path to folder with images'''
    list_of_arr, detections = camdes.file_to_descriptor(folder_path)
    descriptors = [descriptor for arr in list_of_arr for descriptor in arr]
    avg_des = db.get_avg_descriptor(descriptors)
    database.put(name, avg_des)
    database.save()

def labelling():
    '''
    takes pic using camera and labels faces
    '''
    pic = take_picture()
    load_dlib_models()
    list_of_arr, detections = camdes.make_descriptor(pic[np.newaxis, :, :, :])
    descriptors = [descriptor for arr in list_of_arr for descriptor in arr]
    names = []
    for descriptor in descriptors:
        names.append(idf.id_to_faces(database, descriptor))
    #print(names)
    #fig, ax = plt.subplots()
    #for i, detection in enumerate(detections):
    #    print(detection)
    #    draw_labels(ax, fig, detection, names[i])
    #ax.imshow(pic)
    return names

def draw_labels(ax, fig, rect, text):
    # Get the landmarks/parts for the face in box d.
    # Draw the face landmarks on the screen.
    rectangle = Rectangle((rect.left(), rect.top()), rect.width(), rect.height(), linewidth = 1, edgecolor = 'b', facecolor = 'none')
    ax.add_patch(rectangle)
    ax.text(rect.center().x, rect.center().y, text, ha = "center", va = "center", color="#FFFFFF")
