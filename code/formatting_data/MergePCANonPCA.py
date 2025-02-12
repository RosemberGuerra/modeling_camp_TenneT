from pathlib import Path
import pandas as pd


input_file_1 = Path("../../data/kmeans_result/cyfeatures_TA_10clusters.csv")
input_file_2 = Path("../../data/kmeans_result/cyfeatures_TA_8clusters.csv")
output_file = Path("../../data/kmeans_result/cyfeatures_combined.csv")


df1 = pd.read_csv(input_file_1)
df2 = pd.read_csv(input_file_2)

df1 = df1.drop(columns=['is_centroid'])
df2 = df2.drop(columns=['is_centroid'])

df1 = df1.drop(columns=['cluster'])
df2 = df2.drop(columns=['cluster'])

print(df1.tail(20))
print(df2.tail(20))

# Identify overlapping columns excluding 'cy_id'
common_columns = df1.columns[df1.columns != 'cy_id'].intersection(df2.columns)

# Rename overlapping columns in both dataframes
df1_renamed = df1.rename(columns={col: col + '_nonpca' for col in common_columns})
df2_renamed = df2.rename(columns={col: col + '_pca' for col in common_columns})

# Perform the merge on 'cy_id'
df_merged = pd.merge(df1_renamed, df2_renamed, on='cy_id')
print(df_merged.columns)

# Move 'selected_nonpca' to the second last position
cols = df_merged.columns.tolist()
selected_col = 'selected_nonpca'
cols.remove(selected_col)
cols.insert(-1, selected_col)
df_merged = df_merged[cols]


print(df_merged.columns)
print(df_merged.tail(20))

df_merged.to_csv(output_file, index=False)
