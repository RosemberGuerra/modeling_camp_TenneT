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
input_file = Path("../../data/features_output_selection_Years.csv")

# Load data
df = pd.read_csv(input_file)

# Remove column
if 'ModelName_1' in df.columns:
    df.drop(columns='ModelName_1', inplace=True)

# sub set selection for target year: 2050
# select years: form 2036 to 2065
years_2036__2065 = [i for i in range(2036, 2066)]
df_target_2050 = df[df['Year'].isin(years_2036__2065)]

# Drop NaN values
l_init = len(df_target_2050)
df_target_2050 = df_target_2050.dropna()
print("{}/{} rows left without NaN values".format(len(df_target_2050), l_init))

# Normalize feature values
df_scaled = pd.DataFrame(StandardScaler().fit_transform(df_target_2050[df_target_2050.columns[2:]]),
                         columns=df_target_2050.columns[2:])

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
    sum_of_distances_squared.append(clustering_model.inertia_)

# Plot figure showing the sum of squared distances for the different number of clusters
# From this figure, the best number of clusters to use can be determined.
plt.plot(range(2, num_clusters+1), sum_of_distances_squared, 'o')
plt.xticks(range(2, num_clusters+1))
plt.title("No PCA")
plt.xlabel("Number of clusters")
plt.ylabel("Sum of squared distances")
plt.savefig("../../data/kmeans_result/cyfeatures_number_of_clusters_graph.png")
plt.close()
