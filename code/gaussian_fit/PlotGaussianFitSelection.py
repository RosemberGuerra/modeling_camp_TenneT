from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis


# Settings.
plotx = 'PC1'
ploty = 'PC2'
input_file = Path("../../data/PCA_result/pca_results_gaussianselected.csv")


df = pd.read_csv(input_file)


plt.figure()
plt.scatter(df[plotx], df[ploty], color='gray', alpha=0.5, label='Non-selected')
plt.scatter(df.loc[df['selected'], plotx], df.loc[df['selected'], ploty], color='red', label='Selected')
plt.xlabel(plotx)
plt.ylabel(ploty)
plt.legend()
plt.title(f'{ploty} vs {plotx} with Selected Points Highlighted')
plt.show()
