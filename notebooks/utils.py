# Utility script for Jupiter notebooks

# General imports
import os
import numpy as np
import scipy as sp
import sklearn as sk
import tqdm.notebook as tqdm
import cv2
import pickle

# Ploting
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

plt.rcParams["text.usetex"] = True

# Classifiers
from sklearn.dummy import DummyClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Metrics
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import chi2_kernel

# Print versions
print("Imported libraries:")
print(f"\t- Numpy version: {np.__version__}")
print(f"\t- OpenCV version: {cv2.__version__}")
print(f"\t- SciKit-Learn version: {sk.__version__}")
print(f"\t- SciPy version: {sp.__version__}")

print("\nImported functions:")
print("\t- load_images()")
print("\t- compute_descriptors()")
print("\t- intersection_kernel()")
print("\t- normalized_histogram()")
print("\t- tfidf()")
print("\t- kcb()")
print("\t- unc()")
print("\t- pla()")
print("\t- pyramid_histogram()")


# Images loading function
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


# Feature extraction function
def compute_descriptors(images, sift, grid=False, spacing=8):
	"""Compute the SIFT descriptors of a set of images
 
	Parameters
	----------
	images : list
		List of images
	sift : cv2.xfeatures2d_SIFT
		SIFT object
	grid : bool, optional
		Whether to use a grid for dense sampling or not, by default False
	spacing : int, optional
		Spacing step between any two keypints of the grid, by default 8
		Ignored if grid is False
	
	Returns
	-------
	all_descriptors : np.array
		Descriptors of all images returned as a single array
	per_image_descriptors : list
		Descriptors of each image returned as a list of arrays
	"""
	
	# Initialize variables
	all_descriptors = []
	per_image_descriptors = []
	for image in tqdm.tqdm(images, desc="Computing descriptors", leave=False):
		if grid:
			height, width = image.shape[:2]
			keypoints = [
				cv2.KeyPoint(float(x), float(y), float(spacing))
				for y in range(spacing, height, spacing)
				for x in range(spacing, width, spacing)
			]
			_, image_descriptors = sift.compute(image, keypoints)
		else:
			_, image_descriptors = sift.detectAndCompute(image, None)

		image_descriptors = np.array(image_descriptors)
		all_descriptors.extend(image_descriptors)
		per_image_descriptors.append(image_descriptors)
  
	return np.array(all_descriptors), per_image_descriptors
  

# Histogram intersection kernel function
def intersection_kernel(X, Y):
	"""Compute the Gram matrix of the histogram intersection kernel given two sets of
	images represented as histograms

	Parameters
	----------
	X : np.array
		Set of images represented as histograms
	Y : np.array
		Set of images represented as histograms
  
	Returns
	-------
	kernel : np.array
		Gram matrix of the histogram intersection kernel
	"""

	kernel = np.zeros((X.shape[0], Y.shape[0]))
	for i in range(X.shape[0]):
		for j in range(Y.shape[0]):
			intersection = np.minimum(X[i], Y[j])
			kernel[i, j] = np.sum(intersection)

	return kernel


# Image representation functions


# Normalized histogram representation
def normalized_histogram(images, sift, kmeans, grid=False, spacing=8):
	"""Compute the normalized histogram representation of a set of images
	
	Parameters
	----------
	images : list
		List of images
	sift : cv2.xfeatures2d_SIFT
		SIFT object
	kmeans : sklearn.cluster.KMeans
		KMeans object
	grid : bool, optional
		Whether to use a grid for dense sampling or not, by default False
	spacing : int, optional
		Spacing step between any two keypints of the grid, by default 8
		Ignored if grid is False
	
	Returns
	-------
	histogram_representation : np.array
		Normalized histogram representation of the images
	"""
	
	# Initialize variables
	histogram_representation = []
	K = kmeans.n_clusters

	# For each image, compute the histogram representation
	for image in tqdm.tqdm(
		images, 
	 	desc="Computing histogram representations", 
	  	leave=False
	):
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
		histogram = np.bincount(document_words, minlength=K)
		histogram = histogram / np.sum(histogram)
		histogram_representation.append(histogram)

	return np.array(histogram_representation)


# TF-IDF representation
def tfidf(images, sift, kmeans, grid=False, spacing=8):
	"""Compute the TF-IDF representation of a set of images
 
	Parameters
	----------
	images : list
		List of images
	sift : cv2.xfeatures2d_SIFT
		SIFT object
	kmeans : sklearn.cluster.KMeans
		KMeans object
	grid : bool, optional
		Whether to use a grid for dense sampling or not, by default False
	spacing : int, optional
		Spacing step between any two keypints of the grid, by default 8
		Ignored if grid is False
	
	Returns
	-------
	tfidf_representation : np.array
		TF-IDF representation of the images
	"""

	# Initialize variables
	tfidf_representation = []
	K = kmeans.n_clusters
	D = len(images)			# D: total number of documents
	N = np.zeros(K)  		# N[i]: number of documents containing word i
	n = []  				# n[i,d]: occurences of word i in document d
	w = []  				# w[d]: total number of words in document d

	# For each image, fill the variables
	for image in tqdm.tqdm(
		images,
		desc="Computing TF-IDF representations",
		leave=False
	):
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
	 
		document_words = kmeans.predict(descriptors)
		histogram = np.bincount(document_words, minlength=K)
		unique_document_words = set(document_words)
		for word in unique_document_words: N[word] += 1
		n.append(histogram)
		w.append(len(descriptors))

	n = np.array(n)
	w = np.array(w)

	# For each image, compute the tfidf representation
	for d in range(D): # Loop over documents
		image_tfidf = np.zeros(K)
		for i in range(K): # Loop over words
			image_tfidf[i] = (n[d, i] / (w[d] + 1e-6)) * np.log(D / (N[i] + 1e-6))

		tfidf_representation.append(image_tfidf)		

	return np.array(tfidf_representation)



# Kernel codebook representation
def kcb(images, sift, kmeans, radius, grid=False, spacing=8, sigma=200):
	"""Compute the kernel codebook representation of a set of images
 
	Parameters
	----------
	images : list
		List of images
	sift : cv2.xfeatures2d_SIFT
		SIFT object
	kmeans : sklearn.cluster.KMeans
		KMeans object
	radius : np.array
		Array of radius for each cluster
	grid : bool, optional
		Whether to use a grid for dense sampling or not, by default False
	spacing : int, optional
		Spacing step between any two keypints of the grid, by default 8
		Ignored if grid is False
	sigma : int, optional
		Standard deviation of the RBF kernel, by default 200
  
	Returns
	-------
	kcb_representation : np.array
		Kernel codebook representation of the images
	"""

	# Initialize variables
	kcb_representation = []
	K = kmeans.n_clusters
	centroids = kmeans.cluster_centers_
 
	# For each image, compute the kernel codebook representation
	for image in tqdm.tqdm(
		images,
		desc="Computing KCB representations",
		leave=False
	):
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

		gaussian_kernel = rbf_kernel(centroids, descriptors, gamma=1 / (2*sigma**2))
		distances = np.array([np.linalg.norm(descriptors - word, axis=1) for word in centroids])
		for kidx in range(K):
			distances[kidx][distances[kidx] > radius[kidx]] = 0
			distances[kidx][distances[kidx] != 0] = 1
		gaussian_kernel = gaussian_kernel * distances
		image_kcb = np.sum(gaussian_kernel, axis=1)
		image_kcb /= np.sum(image_kcb)
		kcb_representation.append(image_kcb)
  
	return np.array(kcb_representation)


# Codeword uncertainty representation
def unc(images, sift, kmeans, radius, grid=False, spacing=8, sigma=200):
	"""Compute the codeword uncertainty representation of a set of images
	
	Parameters
	----------
	images : list
		List of images
	sift : cv2.xfeatures2d_SIFT
		SIFT object
	kmeans : sklearn.cluster.KMeans
		KMeans object
	radius : np.array
		Array of radius for each cluster
	grid : bool, optional
		Whether to use a grid for dense sampling or not, by default False
	spacing : int, optional
		Spacing step between any two keypints of the grid, by default 8
		Ignored if grid is False
	sigma : int, optional
		Standard deviation of the RBF kernel, by default 200
	
	Returns
	-------
	unc_representation : np.array
		Codeword uncertainty representation of the images
	"""

	# Initialize variables
	unc_representation = []
	K = kmeans.n_clusters
	centroids = kmeans.cluster_centers_
 
	# For each image, compute the kernel codebook representation
	for image in tqdm.tqdm(
		images,
		desc="Computing UNC representations",
		leave=False
	):
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
	 
		n = len(descriptors)
		gaussian_kernel = rbf_kernel(centroids, descriptors, gamma = 1 / (2*sigma**2))
		distances = np.array([np.linalg.norm(descriptors - word, axis=1) for word in centroids])
		for kidx in range(K):
			distances[kidx][distances[kidx] > radius[kidx]] = 0
			distances[kidx][distances[kidx] != 0] = 1
		gaussian_kernel = gaussian_kernel * distances
		kernel_normlization = np.sum(gaussian_kernel, axis=0)
		gaussian_kernel = gaussian_kernel / (kernel_normlization + 1e-6)
		image_unc = np.sum(gaussian_kernel, axis=1)
		image_unc = image_unc / np.sum(image_unc)
		unc_representation.append(image_unc)
  
	return np.array(unc_representation)


# Codeword plausibility representation
def pla(images, sift, kmeans, radius, grid=False, spacing=8, sigma=200):
	"""Compute the codeword plausibility representation of a set of images
 
	Parameters
	----------
	images : list
		List of images
	sift : cv2.xfeatures2d_SIFT
		SIFT object
	kmeans : sklearn.cluster.KMeans
		KMeans object
	radius : np.array
		Array of radius for each cluster
	grid : bool, optional
		Whether to use a grid for dense sampling or not, by default False
	spacing : int, optional
		Spacing step between any two keypints of the grid, by default 8
		Ignored if grid is False
	sigma : int, optional
		Standard deviation of the RBF kernel, by default 200
  
	Returns
	-------
	pla_representation : np.array
		Codeword plausibility representation of the images
	"""
	
	# Initialize variables
	pla_representation = []
	K = kmeans.n_clusters
	centroids = kmeans.cluster_centers_

	# For each image, compute the kernel codebook representation
	for image in tqdm.tqdm(
		images,
		desc="Computing PLA representations",
		leave=False
	):
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
	 
		gaussian_kernel = rbf_kernel(centroids, descriptors, gamma = 1 / (2*sigma**2))
		distances = np.array([np.linalg.norm(descriptors - word, axis=1) for word in centroids])
		mask = np.zeros((len(centroids), len(descriptors)))
		for i in range(len(centroids)): mask[i, np.argmin(distances[i])] = 1
		gaussian_kernel = gaussian_kernel * mask
		image_pla = np.sum(gaussian_kernel, axis=1)
		image_pla = image_pla / np.sum(image_pla)
		pla_representation.append(image_pla)
  
	return np.array(pla_representation)


# Spatial pyramid matching histogram representation
def pyramid_histogram(images, sift, kmeans, levels=[0, 1, 2]):
	"""Compute the spatial pyramid histogram representation of a set of images
	
	Parameters
	----------
	images : list
		List of images
	sift : cv2.xfeatures2d_SIFT
		SIFT object
	kmeans : sklearn.cluster.KMeans
		KMeans object
	levels : list, optional
		List of levels of the pyramid, by default [0, 1, 2]
  
	Returns
	-------
	pyramid_histogram_representation : np.array
		Spatial pyramid histogram representation of the images
	"""
	
	# Initialize variables
	pyramid_histogram_representation = []
	K = kmeans.n_clusters
	spacing = 8 # pixels
 
	# For each image, compute the pyramid histogram representation
	for image in tqdm.tqdm(
		images,
		desc="Computing pyramid histogram representations",
		leave=False
	):
	
		# Image parameters and descriptors
		height, width = image.shape[:2]
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
					cell_histogram = np.bincount(visual_words, minlength=K)

					# Multiply elementwise by the level weights
					cell_histogram = w[l] * cell_histogram

					# Append to histogram
					histogram.extend(cell_histogram) 

		# Normalize and append histogram
		histogram = np.array(histogram)
		histogram = histogram / np.sum(histogram)
		pyramid_histogram_representation.append(histogram)
  
	return np.array(pyramid_histogram_representation)