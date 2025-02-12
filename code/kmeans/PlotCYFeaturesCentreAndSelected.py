from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

n_clusters = 4
print("Making plots for {} clusters".format(n_clusters))
input_file = Path("../../data/kmeans_result/cyfeatures_{}clusters_PCA.csv".format(n_clusters))

# Load data
df = pd.read_csv(input_file)

# Convert 'cluster' to a categorical variable for coloring.
df['cluster'] = df['cluster'].astype(str)

# Create a consistent color palette for clusters.
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

    # Plot points with selected == False (round marker, lower zorder).
    sns.scatterplot(
        data=df[(df['selected'] == False) & (df['is_centroid'] == False)], x=plotx, y=ploty, hue='cluster',
        palette=palette, marker='o', s=50, edgecolor='none', facecolor='none', linewidth=2, alpha=0.7, zorder=1
    )

    # Plot points with is_selected == True (square marker, lower zorder).
    sns.scatterplot(
        data=df[df['selected'] == True], x=plotx, y=ploty, hue='cluster',
        palette=palette, marker='s', s=50, edgecolor='black', alpha=1, zorder=2
    )

    # Plot centroids (cross marker, higher zorder).
    sns.scatterplot(
        data=df[df['is_centroid'] == True], x=plotx, y=ploty, hue='cluster',
        palette=palette, marker='X', s=150, edgecolor='black', alpha=1, zorder=3, legend=False
        # Disable duplicate legend
    )

    # Plot settings.
    plt.title(f"Normalized {ploty} vs {plotx} Colored by Cluster (Centroids Marked)")
    plt.xlabel("Normalized "+plotx)
    plt.ylabel("Normalized "+ploty)

    # Sort legend labels
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.legend(handles, labels, title="Cluster", bbox_to_anchor=(1.02, 1), loc='upper left')

    # Save the plots
    cur_output_file = Path("../../data/kmeans_result/clusterplot_{}clusters_{}_vs_{}.png"
                .format(n_clusters, plotx.replace(" ","_").replace("#", "N").replace("$", ""),
                                    ploty.replace(" ","_").replace("#", "N").replace("$", "")))
    
    print(f"Saving: {cur_output_file}")
    plt.savefig(cur_output_file)
    plt.close()
