# check synthetic data
import numpy as np
import pandas as pd

from src.year_selection import climate_year_selection as cys

# load synthetic data
synthetic_data = pd.read_csv('data/synthetic/reduced_t_data.csv')

# check the data
print(synthetic_data.head())
climate_year = synthetic_data['cy_id']
print(np.unique(climate_year))

# data without years
data_to_run = synthetic_data.drop(columns=['cy_id'])
print(data_to_run.head())
# apply year selection
n_clusters = 3

data_pca = cys.apply_pca(data_to_run.values, explained_variance=0.9)
print(data_pca)

# apply k-medoids clustering
kmedoids_labels, medoids = cys.kmedoid_climate_year(data_pca, n_clusters)

# check the result
print(medoids)
#plot the result
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 5))
plt.scatter(data_pca[:, 0], data_pca[:, 1])
plt.scatter(data_pca[medoids, 0], data_pca[medoids, 1], c='r')
plt.show()