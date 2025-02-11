import pandas as pd
import src.year_selection.climate_year_selection as cys
import seaborn as sns

# import data 
df = pd.read_csv('data/cyfeatures_001.csv')
print(df.head())    
# select year

df_to_run = df.drop(columns=['cy_id']).to_numpy()

n_clusters = 30
c_years = cys.kmedoid_climate_years(df_to_run, n_clusters)
print(c_years)

# plot the results
import matplotlib.pyplot as plt
import numpy as np

# plot the results
print(df_to_run[c_years[1],:])

print(df.iloc[c_years[1]])    # Returns a Series if a single row
# Plot histograms of the F1 with density
plt.hist(df['F1'], bins=20, alpha=0.5, color='blue', label='All Data', density=True)
plt.hist(df.iloc[c_years[1]]['F1'], bins=10, alpha=0.5, color='red', label='Selected Years', density=True)

# Add density lines
sns.kdeplot(df['F1'], color='blue', label='All Data', linewidth=2)
sns.kdeplot(df.iloc[c_years[1]]['F1'], color='red', label='Selected Years', linewidth=2)

plt.legend(loc='upper right')
plt.xlabel('F1')
plt.ylabel('Density')
plt.title('Histogram of F1')
plt.show()

# plot boxplot of the variables for the selected years side by side with the original data
# Create a DataFrame with the selected years

# Ensure df_selected is a DataFrame, not a Series
df_selected = df.iloc[c_years[1]].copy()  

# Add a column to each DataFrame to indicate the source
df['source'] = 'All Data'
df_selected['source'] = 'Selected Years'

# Concatenate the DataFrames
df_combined = pd.concat([df, df_selected])

# Drop 'cy_id' if it exists
df_combined = df_combined.drop(columns=['cy_id'], errors='ignore')

# Convert to long format
df_long = df_combined.melt(id_vars="source", var_name="Variable", value_name="Value")

# Create the boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x="Variable", y="Value", hue="source", data=df_long)
plt.title("Comparison of Variables Between Two Datasets")
plt.xticks(rotation=45)  # Rotate labels for better readability if needed
plt.show()
