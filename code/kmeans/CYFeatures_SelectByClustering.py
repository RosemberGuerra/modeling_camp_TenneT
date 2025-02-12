from pathlib import Path
import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Settings.
method = 'kmedoids'   # Should be chosen from ['kmeans', 'kmedoids'].
num_clusters = 20   # Number of clusters to create.
num_rows = 30   # Number of representative rows to select.
input_file = Path("../../data/PECDc4.2_NLregion_TAfeatures_Vu.csv")

# Load data
df = pd.read_csv(input_file)

# Drop NaN values
l_init = len(df)
df.dropna(inplace=True)
print("{}/{} rows left without NaN values".format(len(df), l_init))

# Normalize feature values
df_scaled = pd.DataFrame(StandardScaler().fit_transform(df[df.columns[1:]]), columns=df.columns[1:])

# Apply Principal Component Analysis
N_pca_components = 5
pca = PCA(n_components=N_pca_components)
pca_result = pca.fit_transform(df_scaled)
print('Cumulative variance explained by {} principal components: {:.2%}'
      .format(N_pca_components, np.sum(pca.explained_variance_ratio_)))

indices = ["PC_{}".format(c) for c in range(1,N_pca_components+1)]
dataset_pca = pd.DataFrame(np.abs(pca.components_), columns=df_scaled.columns, index=indices)
#print(dataset_pca)

# Add the cy_id column back
df_scaled.insert(loc=0, column=df.columns[0], value=df[df.columns[0]])

"""
with pd.option_context('display.max_rows', 100,
                       'display.min_rows', 50,
                       'display.max_columns', None,
                       'display.width', 1000,
                       'display.precision', 8,
                       ):
    print(df)
""";

first_column_is_string = df.iloc[:, 0].apply(lambda x: isinstance(x, str)).all()
print(f"First column contains only strings: {first_column_is_string}")

first_column_unique = df.iloc[:, 0].nunique() == len(df)
if first_column_unique:
    print("First column contains unique values.")
else:
    warnings.warn("First column DOES NOT contain unique values.")

sum_of_distances_squared = []
for N in range(2, num_clusters+1):
    if method == 'kmeans':
        clustering_model = KMeans(n_clusters=N, random_state=42, n_init=10)
    elif method == 'kmedoids':
        clustering_model = KMedoids(n_clusters=N, random_state=42)
    else:
        print("Clustering method not recognized: {}!".format(method))

    # Fit the model on the data, add the cluster assignments to a new column and get cluster centers.
    features_df = df_scaled.iloc[:, 1:].copy()
    clustering_model.fit(features_df)
    df_N = df_scaled.copy()
    df_N['cluster'] = clustering_model.predict(features_df)
    centroids = clustering_model.cluster_centers_
    sum_of_distances_squared.append(clustering_model.inertia_)

    # Find the index of the closest row to each centroid.
    closest_to_centroid_indices = df_N.groupby('cluster').apply(
        lambda x: ((x.iloc[:, 1:-1] - centroids[x.name]) ** 2).sum(axis=1).idxmin()
    ).values

    df_N['is_centroid'] = df_N.index.isin(closest_to_centroid_indices)

    # Select num_rows closest points across clusters.
    df_N['distance_to_centroid'] = df_N.apply(lambda row: ((row.iloc[1:-2] - centroids[row.cluster]) ** 2).sum(), axis=1)
    selected_indices = df_N.nsmallest(num_rows, 'distance_to_centroid').index
    df_N['selected'] = df_N.index.isin(selected_indices)

    df_N.drop(columns=['distance_to_centroid'], inplace=True)
    output_file = Path("../../data/kmeans_result/cyfeatures_TA_{}clusters.csv".format(N))
    df_N.to_csv(output_file, index=False)

# Plot figure showing the sum of squared distances for the different number of clusters
# From this figure, the best number of clusters to use can be determined.
plt.plot(range(2, num_clusters+1), sum_of_distances_squared, 'o')
plt.xticks(range(2, num_clusters+1))
plt.xlabel("Number of clusters")
plt.ylabel("Sum of squared distances")
plt.savefig("../../data/kmeans_result/cyfeatures_TA_number_of_clusters_graph.png")
plt.close()
