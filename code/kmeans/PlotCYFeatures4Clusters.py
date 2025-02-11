from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

input_file = Path("../../data/cyfeatures_002_4clusters.csv")
# input_file = Path("../../data/PCA_result/pca_results_clusters.csv")

plotx = 'F1'
ploty = 'F2'
# plotx = 'PC1'
# ploty = 'PC2'

df = pd.read_csv(input_file)

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

input_file = Path("../../data/cyfeatures_001_4clusters.csv")
# input_file = Path("../../data/PCA_result/pca_results_clusters.csv")

plotx = 'F1'
ploty = 'F2'
# plotx = 'PC1'
# ploty = 'PC2'

df = pd.read_csv(input_file)

# Convert 'cluster' to a categorical variable for coloring.
df['cluster'] = df['cluster'].astype(str)

# Create a consistent color palette for clusters.
unique_clusters = df['cluster'].unique()
if len(unique_clusters) <= 20:
    palette = dict(zip(unique_clusters, sns.color_palette("tab10", len(unique_clusters))))
else:
    palette = dict(zip(unique_clusters, (sns.color_palette("tab20", 20) + sns.color_palette("tab10", len(unique_clusters) - 20) )))

# Create the figure.
plt.figure(figsize=(10, 6))

# Plot points with selected == False (round marker, lower zorder).
sns.scatterplot(
    data=df[(df['selected'] == False) & (df['is_centroid'] == False)], x=plotx, y=ploty, hue='cluster',
    palette=palette, marker='o', s=50, edgecolor='black', alpha=0.7, zorder=1
)

# Plot points with is_selected == True (square marker, lower zorder).
sns.scatterplot(
    data=df[df['selected'] == True], x=plotx, y=ploty, hue='cluster',
    palette=palette, marker='s', s=50, edgecolor='black', alpha=1, zorder=2
)

# Plot centroids (cross marker, higher zorder).
sns.scatterplot(
    data=df[df['is_centroid'] == True], x=plotx, y=ploty, hue='cluster',
    palette=palette, marker='X', s=150, edgecolor='black', alpha=1, zorder=3, legend=False  # Disable duplicate legend
)

# Manually add legend.
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, labels, title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')

# Plot settings.
plt.title(f"{ploty} vs {plotx} Colored by Cluster (Centroids Marked)")
plt.xlabel(plotx)
plt.ylabel(ploty)

# Show the plot.
plt.show()
