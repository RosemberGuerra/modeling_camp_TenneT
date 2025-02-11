# script to add the methods to select the climate year
# method: k-means clustering

# k mean with sklearn
from sklearn.cluster import KMeans
import numpy as np

def kmeans_climate_year_selection(data, n_clusters):
    # data: numpy array of shape (n_samples, n_features)
    # n_clusters: number of clusters to form
    # return: numpy array of shape (n_samples,)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    return kmeans.labels_

