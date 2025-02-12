from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n_clusters = 6
print(f"Making plots for {n_clusters} clusters")
input_file = Path(f"../../data/kmeans_result/normed/cyfeatures_TA_{n_clusters}clusters.csv")

# Load data
df = pd.read_csv(input_file)

# Print column numbers and names
for idx, col in enumerate(df.columns):
    print(f"Column {idx}: {col}")
print()

column_number = 1  # Ensure this column contains numerical data
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
hist_all, bin_edges = np.histogram(all_values, bins=bins)  # Compute histogram for all data
hist_selected, _ = np.histogram(selected_values, bins=bin_edges)  # Compute for selected data

# Scale selected histogram to match total count
scale_factor = hist_all.sum() / hist_selected.sum() if hist_selected.sum() > 0 else 1
hist_selected_scaled = hist_selected * scale_factor

# Plot histograms
plt.hist(all_values, bins=bin_edges, alpha=0.5, label=df.columns[column_number], color='blue')
plt.hist(selected_values, bins=bin_edges, alpha=0.5, weights=[scale_factor] * len(selected_values), 
         label='Selected Data (Scaled)', color='red')

# Labels and legend
plt.xlabel(df.columns[column_number])
plt.ylabel('Frequency')
plt.title(f'Histogram of {df.columns[column_number]}')
plt.legend()

# Show plot
plt.show()
