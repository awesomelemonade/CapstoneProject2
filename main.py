import databasing as db
import Camera_to_Descriptor as camdes
import id_to_faces as idf
import clustering

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