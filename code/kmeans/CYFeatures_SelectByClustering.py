from pathlib import Path
import pandas as pd
import warnings
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

# Settings.
method = 'kmedoids'   # Should be chosen from ['kmeans', 'kmedoids'].
num_clusters = 4   # Number of clusters to create.
num_rows = 30   # Number of representative rows to select.
input_file = Path("../../data/cyfeatures_002.csv")
output_file = Path("../../data/cyfeatures_002_4clusters.csv")

df = pd.read_csv(input_file)
df.dropna(inplace=True)

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
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    features_df = df.iloc[:, 1:].copy()

    # Fit the KMeans model on the data, add the cluster assignments to a new column and get cluster centers.
    kmeans.fit(features_df)
    df['cluster'] = kmeans.predict(features_df)
    centroids = kmeans.cluster_centers_

if method == 'kmedoids':
    kmedoids = KMedoids(n_clusters=num_clusters, random_state=42)
    features_df = df.iloc[:, 1:].copy()

    # Fit the KMeans model on the data, add the cluster assignments to a new column and get cluster centers.
    kmedoids.fit(features_df)
    df['cluster'] = kmedoids.predict(features_df)
    centroids = kmedoids.cluster_centers_

# Find the index of the closest row to each centroid.
closest_to_centroid_indices = df.groupby('cluster').apply(
    lambda x: ((x.iloc[:, 1:-1] - centroids[x.name]) ** 2).sum(axis=1).idxmin()
).values

df['is_centroid'] = df.index.isin(closest_to_centroid_indices)

# Select num_rows closest points across clusters.
df['distance_to_centroid'] = df.apply(lambda row: ((row.iloc[1:-2] - centroids[row.cluster]) ** 2).sum(), axis=1)
selected_indices = df.nsmallest(num_rows, 'distance_to_centroid').index
df['selected'] = df.index.isin(selected_indices)

df.drop(columns=['distance_to_centroid'], inplace=True)
df.to_csv(output_file, index=False)

print(df.tail(20))
