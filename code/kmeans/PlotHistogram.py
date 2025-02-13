from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Settings
n_clusters = 4   # Determines the data set used.
column_number = 8   # This column will be used for the histogram
use_density = False   # Normalization is used if use_density is True
bins = 16

# Select the input file 
print(f"Making plots for {n_clusters} clusters\n")
input_file = Path(f"../../data/kmeans_result/cyfeatures_{n_clusters}clusters_new.csv")

# Load data
df = pd.read_csv(input_file)

# Print column numbers and names and print selected column
for idx, col in enumerate(df.columns):
    print(f"Column {idx}: {col}")
print()
print(f"Column selected: {column_number}: {df.columns[column_number]}\n")
# print(df.iloc[:, column_number])

# Column 'selected' should exist and contain bools
if 'selected' in df.columns:
    df['selected'] = df['selected'].astype(bool)
else:
    raise KeyError("The 'selected' column is missing from the DataFrame.")

# Extract data
all_values = df.iloc[:, column_number]
selected_values = df[df['selected']].iloc[:, column_number]

# Create the plot
plt.figure(figsize=(8, 6))

# Plot histograms
plt.hist(all_values, bins=bins, alpha=0.5, label=df.columns[column_number], color='blue', density=use_density)
plt.hist(selected_values, bins=bins, alpha=0.5, label='Selected Data', color='red', density=use_density)

# Labels and legend
plt.xlabel(df.columns[column_number])
if not use_density:
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {df.columns[column_number]}')
else:
    plt.ylabel('Density')
    plt.title(f'Histogram of {df.columns[column_number]} (Normalized)')
plt.legend()

# Show plot
plt.show()
