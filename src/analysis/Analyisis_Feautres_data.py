# Analysis using the features_output_test.csv fle

# Importing the necessary libraries
import pandas as pd
import numpy as np
import src.year_selection.climate_year_selection as cys

# Load the features_output_test.csv file
features_data = pd.read_csv("data/PECDc4.2_NLregion_TAfeatures_Vu_Years.csv")

features_data.info()
features_data.head()
features_data.shape
# sub set selection for taget year: 2050
# select years: form 2036 to 2065

years_2036__2065 = [i for i in range(2036, 2066)]

df_target_2050 = features_data[features_data['Year'].isin(years_2036__2065)]
df_target_2050.shape
df_target_2050.head()

# check the data types 
df_target_2050.dtypes

# check the missing values
df_target_2050.isnull().sum()

# Data to use: without Year and cy_id columns
df_target_2050_2use = df_target_2050.drop(columns=['Year', 'cy_id'])
df_target_2050_2use.head()

# apply PCA to the data
df_pca = cys.apply_pca(df_target_2050_2use, 0.95)
df_pca.shape

# apply Kmedoids to the data
n_clusters = 30
cy_kmedoids = cys.kmedoid_climate_years(df_pca, n_clusters)

medoids = cy_kmedoids[1]
# Add cluster labels to the data
df_target_2050['cluster'] = np.nan
for i, cluster in enumerate(cy_kmedoids[0]):
    df_target_2050.iloc[cluster, df_target_2050.columns.get_loc('cluster')] = i

# selected climate years #
selected_cy = df_target_2050.iloc[medoids]
selected_cy.head()

