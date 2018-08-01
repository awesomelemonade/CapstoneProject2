import random
import numpy as np
from PIL import Image

def getImage(filename):
	return np.array(Image.open(filename))
def getDictionary(filename):
	dictionary = {}
	with open(filename) as lines:
		for line in lines:
			split = line.split()
			dictionary[split[0]] = int(split[1])
	return dictionary
def distanceSquared(a, b):
	if a is None or b is None:
		return -1e-20
	return np.sum(np.square(a - b))
def cluster(descriptors, iterations):
	"""
	descriptors: shape=(N, ...)
	iterations: int
	"""
	n = len(descriptors)
	types = dict(zip(range(n), range(n)))
	for iteration in range(iterations):
		# Pick a random node
		node = random.randint(0, n - 1)
		weights = {}
		for i in range(n):
			if not i == node:
				dist = distanceSquared(descriptors[node], descriptors[i])
				if dist < 0.3:
					weights[types[i]] = weights.get(types[i], 0) + 1/dist
		max_weight = max(weights.values())
		candidates = [key for key, value in weights.items() if value == max_weight]
		types[node] = random.choice(candidates)
	return types
def getDescriptors(images):
	face_detect = models["face detect"]
	face_rec_model = models["face rec"]
	shape_predictor = models["shape predict"]
	descriptors = []
	for index, image in zip(indices, images):
		detections = list(face_detect(image))
		if len(detections) > 0:
			shape = shape_predictor(image, detections[0])
			descriptors.append(np.array(face_rec_model.compute_face_descriptor(image, shape)))
		else:
			descriptors.append(None)
	return descriptors

from dlib_models import download_model, download_predictor
download_model()
download_predictor()

from dlib_models import load_dlib_models

# this loads the dlib models into memory. You should only import the models *after* loading them.
# This does lazy-loading: it doesn't do anything if the models are already loaded.
load_dlib_models()

from dlib_models import models

data_folder = "./"
label_file = "/database.txt"
format_string = "{}.JPG"

dictionary = getDictionary(label_file)

indices = []
values = []

for i in range(1, 10):
	currentValue = dictionary[format_string.format(i)]
	if currentValue not in values:
		indices.extend([key for key, value in dictionary.items() if value == currentValue])
		values.append(currentValue)

images = [getImage(data_folder + i) for i in indices]
truths = [dictionary[i] for i in indices]

print("Appending to dictionary")

celebrities = {}

for truth, image in zip(truths, images):
	if not truth in celebrities:
		celebrities[truth] = []
	celebrities[truth].append(image)

print("Retrieving Descriptors")
descriptors = getDescriptors(images)
print(len(images))
print(len(descriptors))

print("Clustering {} images".format(len(images)))
types = cluster(descriptors, len(indices) * 20)

from collections import Counter
counter = Counter(truths)
counter2 = Counter(counter.values())
print(counter2.items())
counter = Counter(types.values())
counter2 = Counter(counter.values())
print(counter2.items())


tracked = []
groups = []

for t in types.values():
	if t not in tracked:
		groups.append([key for key, value in types.items() if value == t])
		tracked.append(t)

#uncomment for jupyter notebook
#%matplotlib notebook

import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows=5, ncols=len(groups))

for index, group in enumerate(groups):
	for index2, file in enumerate(group):
		if index2 < 5:
			axs[index2, index].imshow(images[file])
			axs[index2, index].axis('off')
"""data = []
for i in range(len(images)):
	for j in range(i + 1, len(images)):
		data.append(distanceSquared(descriptors[i], descriptors[j]))

#import matplotlib.pyplot as plt

#fig, ax = plt.subplots()
#ax.hist(np.array(data), bins=50)
"""
