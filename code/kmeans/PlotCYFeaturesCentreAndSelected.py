from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import itertools


n_clusters = 4
print("Making plots for {} clusters".format(n_clusters))
input_file = Path("../../data/kmeans_result/cyfeatures_{}clusters_PCA_new.csv".format(n_clusters))

# Load data
df = pd.read_csv(input_file)

# Convert 'cluster' to a categorical variable for coloring
df['cluster'] = df['cluster'].astype(str)

# Create a consistent color palette for clusters
unique_clusters = df['cluster'].unique()
if len(unique_clusters) <= 20:
    palette = dict(zip(unique_clusters, plt.cm.tab10.colors[:len(unique_clusters)]))
else:
    palette = dict(zip(unique_clusters, (plt.cm.tab20.colors[:20] + plt.cm.tab10.colors[len(unique_clusters) - 20:])))

# Loop over feature combinations to make the different plots
columns = df.columns[1:-3]
column_combs = list(itertools.combinations(columns, 2))
for comb in column_combs:
    plotx = comb[0]
    ploty = comb[1]

    # Create the figure
    plt.figure(figsize=(10, 8))

    # Plot points with selected == False (round marker, lower zorder).
    selected_false = df[(df['selected'] == False) & (df['is_centroid'] == False)]
    for cluster in selected_false['cluster'].unique():
        cluster_data = selected_false[selected_false['cluster'] == cluster]
        plt.scatter(cluster_data[plotx], cluster_data[ploty],
                    color=palette[cluster], marker='o', s=80, edgecolor='none',
                    facecolors=palette[cluster], alpha=0.7, zorder=1)

    # Plot points with is_selected == True (square marker, larger size).
    selected_true = df[df['selected'] == True]
    for cluster in selected_true['cluster'].unique():
        cluster_data = selected_true[selected_true['cluster'] == cluster]
        plt.scatter(cluster_data[plotx], cluster_data[ploty],
                    color=palette[cluster], marker='s', s=80, edgecolor='black',
                    facecolors=palette[cluster], alpha=1, zorder=2)

    # Plot centroids (cross marker, with black edge).
    centroids = df[df['is_centroid'] == True]
    for cluster in centroids['cluster'].unique():
        cluster_data = centroids[centroids['cluster'] == cluster]
        plt.scatter(cluster_data[plotx], cluster_data[ploty],
                    color=palette[cluster], marker='X', s=160, edgecolor='black',
                    facecolors=palette[cluster], alpha=1, zorder=3,
                    label=f"Centroid {cluster}" if cluster not in centroids['cluster'].unique() else None)

    # Plot settings.
    plt.title(f"Normalized {ploty} vs {plotx} Colored by Cluster (Centroids Marked)")
    plt.xlabel("Normalized " + plotx)
    plt.ylabel("Normalized " + ploty)

    # Manually create legend if it's missing or needs to be sorted
    handles, labels = plt.gca().get_legend_handles_labels()
    if not handles:
        # If no handles (legend entries) are found, we add manual ones
        for cluster in df['cluster'].unique():
            handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[cluster], markersize=10))
            labels.append(cluster)
    # Sort legend labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.legend(handles, labels, title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout to make room for the legend
    plt.tight_layout()

    # Save the plots
    cur_output_file = Path("../../data/kmeans_result/clusterplot_{}clusters_{}_vs_{}.png"
                           .format(n_clusters, plotx.replace(" ", "_").replace("#", "N").replace("$", ""),
                                   ploty.replace(" ", "_").replace("#", "N").replace("$", "")))

    print(f"Saving: {cur_output_file}")
    plt.savefig(cur_output_file)
    plt.close()
