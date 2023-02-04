import time
import numpy as np

from evaluation import davies_bouldin_index, silhouette_coefficient, calculate_top_features
from sklearn.metrics import pairwise_distances


def kmeans_plusplus(X_ori, n_clusters, metric='euclidean'):
	"""Computational component for initialization of n_clusters by
	k-means++.

	Parameters
	----------
	X : ndarray of shape (n_samples, n_features)

	n_clusters : int
		The number of seeds to choose.

	Returns
	-------
	centers : ndarray of shape (n_clusters, n_features)
		The inital centers for k-means.

	indices : ndarray of shape (n_clusters,)
		The index location of the chosen centers in the data array X. For a
		given index and center, X[index] = center.
	"""
	X = X_ori.copy()
	n_samples, n_features = X.shape
	centers = np.empty((n_clusters, n_features), dtype=X.dtype)

	# Pick first center randomly and track index of point
	center_id = np.random.randint(n_samples)
	indices = np.full(n_clusters, -1, dtype=int)
	
	centers[0] = X[center_id]
	indices[0] = center_id

	for c in range(1, n_clusters):
		dist = pairwise_distances(X, centers, metric=metric)
		min_dist_to_closest_center = np.min(dist, axis=1)

		best_candidate = np.argmax(min_dist_to_closest_center)
		centers[c] = X[best_candidate]
		indices[c] = best_candidate
	
	return centers, indices



def _kmeans(X_ori, vocabs, n_clusters, init='k-means', metric='euclidean'):

	"""K-Means Clustering Algorithm.

	Parameters
	----------
	X : ndarray of shape (n_samples, n_features)

	n_clusters : int
		The number of seeds to choose.

	init : 'k-means' or 'k-means++'
		Selection for initial centers

	Returns
	-------
	indices : ndarray of shape (n_clusters,)
		The index location of the chosen centers in the data array X. For a
		given index and center, X[index] = center.

	clusters : ndarray of shape (n_samples,)
		Labels of each point

	centers : ndarray of shape (n_clusters, n_features)
		The final centers for k-means.
	"""
	
	X = X_ori.copy()
	start = time.time()
	n_samples, n_features = X.shape

	# Get centers based algorithm (init) used
	if init == 'k-means++':

		# Get centers from k-means++
		centers, indices = kmeans_plusplus(X, n_clusters, metric=metric)
	else:

		# Get centers from random
		indices = np.random.permutation(n_samples)[:n_clusters]
		centers = X[indices]

	# Initialize previous cluster
	previous_clusters = None

	_iter = 1
	while True:

		# Compute all datapoint to current centers using euclidean distance
		# which will resulting a matrix n_samples x n_clusters
		distances = pairwise_distances(X, centers, metric=metric)

		# Assign datapoint cluster which has minimum distance to specific center
		clusters = np.argmin(distances, axis=1)

		# Check current cluster with previuos cluster
		# if cluster elements has no changes, then stop
		if previous_clusters is not None:
			convergent = (previous_clusters == clusters).all()
			if convergent:
				break

		# Update centers based cluster
		centers = np.empty((n_clusters, n_features), dtype=X.dtype)
		for c in range(n_clusters):

			# Compute "mean" of cluster c then assign it
			centers[c] = np.mean(X[clusters == c], axis=0)

		# Set previous cluster as current cluster
		previous_clusters = np.copy(clusters)

		# Increase iteration
		_iter += 1

	exec_time = time.time() - start
	davies = davies_bouldin_index(X_ori, clusters, metric=metric)
	silhouette = silhouette_coefficient(X_ori, clusters, metric=metric)
	features = calculate_top_features(vocabs, centers)

	return {
		'davies': davies,
		'silhouette': silhouette,
		'features': features,
		'init': indices, 
		'cluster': clusters, 
		'centroid': centers, 
		'iter': _iter, 
		'time': exec_time
	}


def kmeans(X, vocabs, n_clusters, init='k-means', n_try=1, metric='euclidean'):
	results = []
	davies = []
	silhouette = []
	for _ in range(n_try):
		result = _kmeans(X, vocabs, n_clusters, init=init, metric=metric)
		davies += [result['davies']]
		silhouette += [result['silhouette']]
		results.append(result)
	davies = [d / max(davies) for d in davies]
	silhouette = [(1 - s) / 2 for s in silhouette]
	mixed = [d + s for d, s in zip(davies, silhouette)]
	return results[np.argmin(mixed)]
