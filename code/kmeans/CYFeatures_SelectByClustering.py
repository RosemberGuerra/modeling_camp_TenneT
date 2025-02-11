from pathlib import Path
import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt

# Settings.
method = 'kmedoids'   # Should be chosen from ['kmeans', 'kmedoids'].
rep_n = 20   # Number of representative rows to select.
# input_file = Path("/home/joris/Datadir/ModelingCamp2025/DatasetReduced/ClimateYearFeatures.csv")
input_file = Path("../../data/PECDv4.2_NLregion_TAfeatures.csv")
output_file = Path("../../data/kmeans_result/cyfeatures_TA.csv")


df = pd.read_csv(input_file)

# Filter out rows with nan values
l_init = len(df)
for column in df.columns[1:]:
    df = df[~np.isnan(df[column])]
print("{}/{} rows left without NaN values".format(len(df), l_init))

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
for N in range(2, rep_n+1):
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=N, random_state=42, n_init=10)
        kmeans_df = df.copy()
        features_df = df.iloc[:, 1:].copy()

        # Fit the KMeans model on the data, add the cluster assignments to a new column and get cluster centers.
        kmeans.fit(features_df)
        sum_of_distances_squared.append(kmedoids.inertia_)
        kmeans_df['cluster'] = kmeans.fit_predict(features_df)
        kmeans_centroids = kmeans.cluster_centers_

        # Find the index of the closest row to each centroid.
        closest_to_centroid_indices = kmeans_df.groupby('cluster').apply(
            lambda x: ((x.iloc[:, 1:-1] - kmeans_centroids[x.name]) ** 2).sum(axis=1).idxmin()
        ).values   # Extract indices as an array.

        # Add a new boolean column marking centroid rows.
        kmeans_df['is_centroid'] = kmeans_df.index.isin(closest_to_centroid_indices)

        #print(kmeans_df.tail(20))  # Display last few rows.
        output_file_N = str(output_file).replace(".csv", "_{}clusters.csv".format(N))
        kmeans_df.to_csv(output_file_N, index=False)


    if method == 'kmedoids':
        kmedoids = KMedoids(n_clusters=N, random_state=42)
        kmedoids_df = df.copy()
        features_df = df.iloc[:, 1:].copy()

        # Fit the KMeans model on the data, add the cluster assignments to a new column and get cluster centers.
        kmedoids.fit(features_df)
        sum_of_distances_squared.append(kmedoids.inertia_)
        kmedoids_df['cluster'] = kmedoids.fit_predict(features_df)
        kmedoids_centroids = kmedoids.cluster_centers_

        # Find the index of the closest row to each centroid.
        closest_to_centroid_indices = kmedoids_df.groupby('cluster').apply(
            lambda x: ((x.iloc[:, 1:-1] - kmedoids_centroids[x.name]) ** 2).sum(axis=1).idxmin()
        ).values   # Extract indices as an array.

        # Add a new boolean column marking centroid rows.
        kmedoids_df['is_centroid'] = kmedoids_df.index.isin(closest_to_centroid_indices)

        #print(kmedoids_df.tail(20))  # Display last few rows.
        output_file_N = str(output_file).replace(".csv", "_{}clusters.csv".format(N))
        kmedoids_df.to_csv(output_file_N, index=False)

plt.plot(range(2, rep_n+1), sum_of_distances_squared, 'o')
plt.xticks(range(2, rep_n+1))
plt.xlabel("Number of clusters")
plt.ylabel("Sum of squared distances")
plt.savefig("../../data/kmeans_result/cyfeatures_TA_number_of_clusters_graph.png")
plt.close()
