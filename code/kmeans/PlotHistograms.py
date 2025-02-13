from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Settings
n_clusters = 4   # Determines the data set used
use_density = False   # Normalization is used if use_density is True
bins = 16
save_plots = True
show_plots = False

# Select the input file 
print(f"Making plots for {n_clusters} clusters\n")
input_file = Path(f"../../data/kmeans_result/cyfeatures_{n_clusters}clusters_new.csv")
# input_file = Path(f"../../data/kmeans_result/cyfeatures_{n_clusters}clusters_PCA_new.csv")

# Load data
df = pd.read_csv(input_file)

# Print column numbers and names and print selected column
for idx, col in enumerate(df.columns):
    print(f"Column {idx}: {col}")
print()
# print(df.iloc[:, column_number])

# Column 'selected' should exist and contain bools
if 'selected' in df.columns:
    df['selected'] = df['selected'].astype(bool)
else:
    raise KeyError("The 'selected' column is missing from the DataFrame.")

# Loop over all but the first and last 3 columns
for cur_col in range(1,len(df.columns)-3):
    # Extract data
    all_values = df.iloc[:, cur_col]
    selected_values = df[df['selected']].iloc[:, cur_col]
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    
    # Plot histograms
    plt.hist(all_values, bins=bins, alpha=0.5, label=df.columns[cur_col], color='blue', density=use_density)
    plt.hist(selected_values, bins=bins, alpha=0.5, label='Selected Data', color='red', density=use_density)

    # Labels and legend
    plt.xlabel(df.columns[cur_col])
    if not use_density:
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {df.columns[cur_col]}')
    else:
        plt.ylabel('Density')
        plt.title(f'Histogram of {df.columns[cur_col]} (Normalized)')
    plt.legend()

    # Save plot
    if save_plots:
        if not use_density:
            cur_outfile = Path("../../data/kmeans_result/histogram_{}clusters_{}.png"
                        .format(n_clusters, df.columns[cur_col]).replace(" ", "_").replace("#", "N").replace("$", "").replace("\\", ""))
            print("Saving: {}".format(cur_outfile))
            plt.savefig(cur_outfile)
        else:
            cur_outfile = Path("../../data/kmeans_result/histogram_normalized_{}clusters_{}.png"
                        .format(n_clusters, df.columns[cur_col]).replace(" ", "_").replace("#", "N").replace("$", "").replace("\\", ""))
            print("Saving: {}".format(cur_outfile))
            plt.savefig(cur_outfile)
    
    # Show plot
    if show_plots:
        plt.show()
    
    plt.close()
