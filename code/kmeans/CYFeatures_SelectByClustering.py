from pathlib import Path
import pandas as pd
import warnings
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids


# Settings.
method = 'kmeans'   # Should be chosen from ['kmeans', 'kmedoids'].
rep_n = 4   # Number of representative rows to select.
# input_file = Path("/home/joris/Datadir/ModelingCamp2025/DatasetReduced/ClimateYearFeatures.csv")
input_file = Path("../../data/cyfeatures_001.csv")
output_file = Path("../../data/cyfeatures_001_clusters.csv")


df = pd.read_csv(input_file)


with pd.option_context('display.max_rows', 100,
                       'display.min_rows', 50,
                       'display.max_columns', None,
                       'display.width', 1000,
                       'display.precision', 8,
                      ):
    print(df)

first_column_is_string = df.iloc[:, 0].apply(lambda x: isinstance(x, str)).all()
print(f"First column contains only strings: {first_column_is_string}")

first_column_unique = df.iloc[:, 0].nunique() == len(df)
if first_column_unique:
    print("First column contains unique values.")
else:
    warnings.warn("First column DOES NOT contain unique values.")


if method == 'kmeans':
    kmeans = KMeans(n_clusters=rep_n, random_state=42, n_init=10)
    kmeans_df = df.copy()
    features_df = df.iloc[:, 1:].copy()

    # Fit the KMeans model on the data, add the cluster assignments to a new column and get cluster centers.
    kmeans.fit(features_df)
    kmeans_df['cluster'] = kmeans.fit_predict(features_df)
    kmeans_centroids = kmeans.cluster_centers_

    # Find the index of the closest row to each centroid.
    closest_to_centroid_indices = kmeans_df.groupby('cluster').apply(
        lambda x: ((x.iloc[:, 1:-1] - kmeans_centroids[x.name]) ** 2).sum(axis=1).idxmin()
    ).values   # Extract indices as an array.

    # Add a new boolean column marking centroid rows.
    kmeans_df['is_centroid'] = kmeans_df.index.isin(closest_to_centroid_indices)

    print(kmeans_df.tail(20))  # Display last few rows.

    kmeans_df.to_csv(output_file, index=False)


if method == 'kmedoids':
    kmedoids = KMedoids(n_clusters=rep_n, random_state=42)
    kmedoids_df = df.copy()
    features_df = df.iloc[:, 1:].copy()

    # Fit the KMeans model on the data, add the cluster assignments to a new column and get cluster centers.
    kmedoids.fit(features_df)
    kmedoids_df['cluster'] = kmedoids.fit_predict(features_df)
    kmedoids_centroids = kmedoids.cluster_centers_

    # Find the index of the closest row to each centroid.
    closest_to_centroid_indices = kmedoids_df.groupby('cluster').apply(
        lambda x: ((x.iloc[:, 1:-1] - kmedoids_centroids[x.name]) ** 2).sum(axis=1).idxmin()
    ).values   # Extract indices as an array.

    # Add a new boolean column marking centroid rows.
    kmedoids_df['is_centroid'] = kmedoids_df.index.isin(closest_to_centroid_indices)

    print(kmedoids_df.tail(20))  # Display last few rows.

    kmedoids_df.to_csv(output_file, index=False)
