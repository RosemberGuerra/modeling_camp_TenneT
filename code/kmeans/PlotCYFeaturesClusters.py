from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


input_file = Path("../../data/cyfeatures_001_clusters.csv")
plotx = 'F2'
ploty = 'F3'

df = pd.read_csv(input_file)


# Convert 'cluster' to a categorical variable for coloring.
df['cluster'] = df['cluster'].astype(str)


# Create a consistent color palette for clusters.
unique_clusters = df['cluster'].unique()
palette = dict(zip(unique_clusters, sns.color_palette("tab10", len(unique_clusters))))

# Create the figure
plt.figure(figsize=(10, 6))

# Plot non-centroids first (lower zorder).
sns.scatterplot(
    data=df[df['is_centroid'] == False], x=plotx, y=ploty, hue='cluster',
    palette=palette, marker='o', s=50, edgecolor='black', alpha=0.7, zorder=1
)

# Plot centroids on top with the same colors (higher zorder).
sns.scatterplot(
    data=df[df['is_centroid'] == True], x=plotx, y=ploty, hue='cluster',
    palette=palette, marker='X', s=150, edgecolor='black', alpha=1, zorder=3, legend=False  # Disable duplicate legend
)

# Plot settings.
plt.title(f"{ploty} vs {plotx} Colored by Cluster (Centroids Marked)")
plt.xlabel(plotx)
plt.ylabel(ploty)
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot.
plt.show()
