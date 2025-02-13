from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

# Settings
n_clusters = 4
save_plots = True
show_plots = False

# Select the input file 
print("Making plots for {} clusters".format(n_clusters))
input_file = Path("../../data/kmeans_result/cyfeatures_{}clusters_PCA_new.csv".format(n_clusters))

# Load data
df = pd.read_csv(input_file)

# Convert 'cluster' to a categorical variable for coloring
df['cluster'] = df['cluster'].astype(str)

# Create a consistent color palette for clusters
unique_clusters = df['cluster'].unique()
if len(unique_clusters) <= 20:
    palette = dict(zip(unique_clusters, sns.color_palette("tab10", len(unique_clusters))))
else:
    palette = dict(zip(unique_clusters, (sns.color_palette("tab20", 20) + sns.color_palette("tab10", len(unique_clusters) - 20) )))

# Loop over feature combinations to make the different plots
columns = df.columns[1:-3]
column_combs = list(itertools.combinations(columns, 2))
for comb in column_combs:
    plotx = comb[0]
    ploty = comb[1]

    # Create the figure
    plt.figure(figsize=(10, 8))

    # Plot non-centroids first (lower zorder)
    sns.scatterplot(
        data=df[df['is_centroid'] == False], x=plotx, y=ploty, hue='cluster',
        palette=palette, marker='o', s=50, edgecolor='black', alpha=0.7, zorder=1
    )

    # Plot centroids on top with the same colors (higher zorder)
    sns.scatterplot(
        data=df[df['is_centroid'] == True], x=plotx, y=ploty, hue='cluster',
        palette=palette, marker='X', s=150, edgecolor='black', alpha=1, zorder=3, legend=False  # Disable duplicate legend
    )

    # Plot settings
    plt.title(f"Normalized {ploty} vs {plotx} Colored by Cluster (Centroids Marked)")
    plt.xlabel("Normalized "+plotx)
    plt.ylabel("Normalized "+ploty)

    # Sort legend labels
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.legend(handles, labels, title="Cluster", bbox_to_anchor=(1.02, 1), loc='upper left')

    # Save plot
    if save_plots:
        xlab = plotx.replace(" ","_").replace("$","").replace("#","N").replace("\\","")
        ylab = ploty.replace(" ","_").replace("$","").replace("#","N").replace("\\","")
        cur_outfile = Path("../../data/kmeans_result/clusterplots/clusterplot_{}clusters_{}_vs_{}.png"
                .format(n_clusters, xlab, ylab))
        plt.savefig(cur_outfile)
    
    # Show plot
    if show_plots:
        plt.show()
    
    plt.close()
