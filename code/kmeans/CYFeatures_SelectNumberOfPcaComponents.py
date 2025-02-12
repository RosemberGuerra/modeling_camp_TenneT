from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Settings.
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
pca = PCA(n_components=10)
pca.fit(df_scaled)

plt.figure(figsize=(10,8))
plt.plot(range(1,11), pca.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
plt.title("Explained Variance by Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.savefig("../../data/kmeans_result/graph2select_number_of_PCA_components.png")
plt.close()
