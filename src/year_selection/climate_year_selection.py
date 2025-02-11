# script to add the methods to select the climate year
# method: k-means clustering

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from pyclustering.cluster.kmedoids import kmedoids
from scipy.spatial.distance import cdist

def apply_pca(X, explained_variance=0.9):
    """
    Standardizes the input data and applies PCA to reduce dimensionality.

    Parameters:
    - X: DataFrame or NumPy array (rows = climate years, columns = climate variables)
    - explained_variance: float, proportion of variance to retain (default 0.95)

    Returns:
    - X_pca: PCA-transformed data
    - pca_model: Fitted PCA model (for inverse transformation if needed)
    - scaler: Fitted StandardScaler (to apply inverse scaling if needed)
    """
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=explained_variance)  # Retain given percentage of variance
    X_pca = pca.fit_transform(X_scaled)

    print(f"PCA Reduced Dimensions: {X_pca.shape[1]} (from {X.shape[1]})")

    return X_pca

def kmedoid_climate_year(X, n_clusters):
    """
    Apply k-medoids clustering to select climate years.

    Parameters:
    - X: DataFrame or NumPy array (rows = climate years, columns = climate variables)
    - n_clusters: int, number of clusters to form

    Returns:
    - labels: Cluster labels for each climate year
    """
    
    # Apply k-medoids clustering
    kmedoids_instance = kmedoids(X, 
                                 initial_index_medoids=np.random.choice(X.shape[0],n_clusters, replace=False))
    kmedoids_instance.process()
    labels = kmedoids_instance.get_clusters()
    medoids = kmedoids_instance.get_medoids()

    return labels, medoids

