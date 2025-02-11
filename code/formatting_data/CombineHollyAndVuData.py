from pathlib import Path
import pandas as pd


input_file_1 = Path("/home/joris/Datadir/ModelingCamp2025/DatasetReduced/reduced_t_data.csv")
input_file_2 = Path("/home/joris/Datadir/ModelingCamp2025/DatasetReduced/NL_ECE3_SP245_features.csv")
output_file = Path("/home/joris/Datadir/ModelingCamp2025/DatasetReduced/cyfeatures_001.csv")


df1 = pd.read_csv(input_file_1)
df2 = pd.read_csv(input_file_2)

df2.rename(columns={'Annual average temperature':'F4', 'Integrated temperature distance':'F5'}, inplace=True)

print(df1.tail(20))
print(df2.tail(20))

# Merge the DataFrames on the 'cy_id' column.
merged_df = pd.merge(df1, df2, on="cy_id", how="inner")


print(merged_df.tail(20))

merged_df.to_csv(output_file, index=False)
