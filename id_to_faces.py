import numpy as np
from collections import Counter
import databasing

def id_to_faces(descriptor_database, descriptor, diff=0.01):
    """Identifies a face from the database when given a descriptor
    Parameters
    ----------
        descriptor_database : DescriptorDatabase
            A database of descriptors for faces, which can be turned into a dictionary
        descriptor: np.ndarray
            A descriptor for a face, of shape (128,)
        diff : float
            A number representing the maximum allowed difference
    Returns
    -------
        face : String
            Returns who the face belongs to
    """
    descriptor_database.load()
    database = descriptor_database.dictionary
    counter = np.zeros(len(database.keys()))
    for d in range(len(names)):
        counter[d] = np.sqrt(np.sum((database[names[d]] - descriptor) ** 2))
    return names[np.argmin(counter)]
