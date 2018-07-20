import random
def distanceSquared(a, b):
	return np.sum(np.square(a - b))
def cluster(images, descriptors, iterations):
	"""
	images: shape=(N, ...)
	descriptors: shape=(N, ...)
	"""
	n = len(images)
	types = zip(range(n), range(n))
	for iteration in range(iterations):
		# Pick a random node
		node = random.randint(n)
		weights = {}
		for i in range(n):
			if not i == node:
				weights[types[i]] = weights.get(types[i], 0) + 1/distanceSquared(descriptors[node], descriptors[i])
		max_weight = max(weights.values())
		candidates = [key for key, value in weight.items() if value == max_weight]
		types[node] = random.choice(candidates)
	return types
