from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Settings.
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
