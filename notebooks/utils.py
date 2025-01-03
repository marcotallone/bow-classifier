# Utility script for Jupiter notebooks

# General imports
import os
import numpy as np
import tqdm.notebook as tqdm
import cv2

# Ploting
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

plt.rcParams["text.usetex"] = True

# Classifiers
import sklearn as sk
from sklearn.dummy import DummyClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Metrics
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import chi2_kernel

# Print versions
print("Imported libraries:")
print(f"\t- Numpy version: {np.__version__}")
print(f"\t- OpenCV version: {cv2.__version__}")
print(f"\t- SciKit-Learn version: {sk.__version__}")

print("\nImported functions:")
print("\t- load_images()")
print("\t- get_histogram()")
print("\t- get_tfidf()")
print("\t- get_pyramid_histogram()")
print("\t- pyramid_kernel()")


# Function to load the images and corresponding labels
def load_images(folder):
	"""Load images from a folder

	Parameters
	----------
	folder : str
		Path to the folder containing the images

	Returns
	-------
	images : list
		List of images
	labels : list
		List of numeric labels corresponding to the images
	classes : list
		List of class names corresponding to the labels
	"""

	images = []
	labels = []
	classes = sorted(os.listdir(folder))
	classes = [class_name for class_name in classes if not class_name.startswith(".")]
	indices = {class_name: i for i, class_name in enumerate(classes)}
	for class_name in tqdm.tqdm(classes, desc="Loading images", leave=False):
		class_folder = os.path.join(folder, class_name)
		if os.path.isdir(class_folder):
			for filename in os.listdir(class_folder):
				image_path = os.path.join(class_folder, filename)
				image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
				if image is not None:
					images.append(image)
					labels.append(indices[class_name])
	return images, labels, classes


# Function to get the normalized histogram representation
def get_histogram(image, sift, kmeans, k, grid=False, spacing=8):
	"""Compute the normalized histogram representation of an image

	Parameters
	----------
	image : np.array
		Image to compute the histogram representation
	sift : cv2.xfeatures2d_SIFT
		SIFT object
	kmeans : cv2.KMeans
		KMeans object
	k : int
		Number of clusters, i.e. the size of the visual vocabulary
	grid : bool, optional
		Whether to use a grid for dense sampling or not, by default False
	spacing : int, optional
		Spacing step between any two keypints of the grid, by default 8
		Ignored if grid is False

	Returns
	-------
	histogram : np.array
		Histogram representation of the image
	"""

	# Compute the descriptors (either with detector or with dense grid)
	if grid:
		height, width = image.shape[:2]
		keypoints = [
			cv2.KeyPoint(float(x), float(y), float(spacing))
			for y in range(spacing, height, spacing)
			for x in range(spacing, width, spacing)
		]
		_, descriptors = sift.compute(image, keypoints)
	else:
		_, descriptors = sift.detectAndCompute(image, None)
	descriptors = np.array(descriptors)

	# Compute the histogram by counting the words in the image
	document_words = kmeans.predict(descriptors)
	histogram = np.bincount(document_words, minlength=k)

	# Normalize the histogram
	histogram = np.array(histogram / len(descriptors))

	return histogram


# Term Frequency - Inverse Document Frequency (TF-IDF) function
def get_tfidf(images, sift, kmeans, k, grid=False, spacing=8):
	"""Compute the TF-IDF representation of a set of images

	Parameters
	----------
	images : list
		List of images
	sift : cv2.xfeatures2d_SIFT
		SIFT object
	kmeans : cv2.KMeans
		KMeans object
	k : int
		Number of clusters, i.e. the size of the visual vocabulary
	grid : bool, optional
		Whether to use a grid for dense sampling or not, by default False
	spacing : int, optional
		Spacing step between any two keypints of the grid, by default 8
		Ignored if grid is False

	Returns
	-------
	tfidf : np.array
		TF-IDF representation of the images
	"""

	# Initialize variables
	D = len(images)	# D      : total number of documents
	N = np.zeros(k)	# N[i]   : number of documents containing word i
	n = []  		# n[i,d] : occurences of word i in document d
	w = []  		# w[d]   : total number of words in document d

	# Fill the variables
	for image in tqdm.tqdm(images, desc="Computing TF-IDF", leave=False):

		if grid:
			height, width = image.shape[:2]
			keypoints = [
				cv2.KeyPoint(float(x), float(y), float(spacing))
				for y in range(spacing, height, spacing)
				for x in range(spacing, width, spacing)
			]
			_, descriptors = sift.compute(image, keypoints)
		else:
			_, descriptors = sift.detectAndCompute(image, None)
		descriptors = np.array(descriptors)

		document_words = kmeans.predict(descriptors)
		histogram = np.bincount(document_words, minlength=k)
		unique_document_words = set(document_words)
		for word in unique_document_words: N[word] += 1

		n.append(histogram)
		w.append(len(descriptors))

	n = np.array(n)
	w = np.array(w)

	# Compute the TF-IDF for each image and word in that image
	tfidf = []	
 
	# Loop over documents
	for d in range(D):
		tfidf_image = np.zeros(k)
  
		#Loop over words
		for i in range(k):
			tfidf_image[i] = (n[d, i] / (w[d] + 1e-6)) * np.log(D / (N[i] + 1e-6))
   
		tfidf.append(tfidf_image)		

	return np.array(tfidf)


# Extended weighted histogram function for the pyramid matching kernel
def get_pyramid_histogram(image, levels, sift, kmeans, k):
	"""Compute the pyramid histogram representation of an image
 
	Parameters
	----------
	image : np.array
		Image to compute the histogram representation
	levels : list
		List of levels of the pyramid
	sift : cv2.xfeatures2d_SIFT
		SIFT object
	kmeans : cv2.KMeans
		KMeans object
	k : int
		Number of clusters, i.e. the size of the visual vocabulary
	
	Returns
	-------
	histogram : np.array
		Histogram representation of the image
	"""
	
	# Image parameters and descriptors
	height, width = image.shape[:2]
	spacing = 8 # pixels
	keypoints = [
		cv2.KeyPoint(float(x), float(y), float(spacing))
		for y in range(spacing, height, spacing)
		for x in range(spacing, width, spacing)
	]
	_, descriptors = sift.compute(image, keypoints)
 
	# Level weigths
	L = levels[-1]
	w = [(0.5 ** (L - l + 1)) for l in levels]
	w[0] = 0.5 ** L

	# Initialize histogram
	histogram = []
 
	# Iterate over levels
	for l in levels:
	
		# Cells parameters
		n_cells = 2 ** l
		cell_height = height / n_cells
		cell_width = width / n_cells

		# Iterate over cells
		for i in range(n_cells):
			for j in range(n_cells):
					   
				# Coordinates
				x_min = i * cell_width
				y_min = j * cell_height
				x_max = (i + 1) * cell_width
				y_max = (j + 1) * cell_height
    
				# Descriptors in the cell
				cell_descriptors = np.array([
					descriptor 
		   			for keypoint, descriptor in zip(keypoints, descriptors) 
			  		if (x_min <= keypoint.pt[0] < x_max 
					and y_min <= keypoint.pt[1] < y_max)
				])
	
				# Cell histogram
				visual_words = kmeans.predict(cell_descriptors)
				cell_histogram = np.bincount(visual_words, minlength=k)

				# Multiply elementwise by the level weights
				cell_histogram = w[l] * cell_histogram

				# Append to histogram
				histogram.extend(cell_histogram) 

	# Normalize histogram
	histogram = np.array(histogram)
	histogram = histogram / np.sum(histogram)
 
	return histogram


# Pyramid matching kernel function
def pyramid_kernel(histogram_set_1, histogram_set_2):
	"""Compute the Gram matrix of the pyramid matching kernel given two sets of
	images represented as extended weighted histograms

	Parameters
	----------
	histogram_set_1 : np.array
		Set of images represented as extended weighted histograms
	histogram_set_2 : np.array
		Set of images represented as extended weighted histograms
  
	Returns
	-------
	kernel : np.array
		Gram matrix of the pyramid matching kernel
	"""

	kernel = np.zeros((len(histogram_set_1), len(histogram_set_2)))
	for i in tqdm.tqdm(range(len(histogram_set_1)), desc="Computing kernel", leave=False):
		for j in range(len(histogram_set_2)):
			intersection = np.minimum(histogram_set_1[i], histogram_set_2[j])
			kernel[i, j] = np.sum(intersection)

	return kernel