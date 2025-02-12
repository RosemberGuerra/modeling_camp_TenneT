from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n_clusters = 6
print(f"Making plots for {n_clusters} clusters")
input_file = Path(f"../../data/kmeans_result/cyfeatures_TA_{n_clusters}clusters.csv")

# Load data
df = pd.read_csv(input_file)

# Print column numbers and names
for idx, col in enumerate(df.columns):
    print(f"Column {idx}: {col}")
print()

column_number = 2  # Ensure this column contains numerical data
print(f"Column selected: {column_number}: {df.columns[column_number]}")
print(df.iloc[:, column_number])

# Ensure 'selected' column exists and is boolean
if 'selected' in df.columns:
    df['selected'] = df['selected'].astype(bool)
else:
    raise KeyError("The 'selected' column is missing from the DataFrame.")

# Extract data
all_values = df.iloc[:, column_number]
selected_values = df[df['selected']].iloc[:, column_number]

# Create histograms
plt.figure(figsize=(8, 6))

# Define bins
bins = 16  # Adjust as needed

# Plot histograms using density=True for normalization
plt.hist(all_values, bins=bins, alpha=0.5, label=df.columns[column_number], color='blue', density=True)
plt.hist(selected_values, bins=bins, alpha=0.5, label='Selected Data', color='red', density=True)

# Labels and legend
plt.xlabel(df.columns[column_number])
plt.ylabel('Density')
plt.title(f'Histogram of {df.columns[column_number]} (Normalized)')
plt.legend()

# Show plot
plt.show()
