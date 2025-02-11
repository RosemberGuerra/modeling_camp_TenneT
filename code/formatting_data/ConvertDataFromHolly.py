from pathlib import Path
import pandas as pd


input_file = Path("/home/joris/Datadir/ModelingCamp2025/DatasetReduced/reduced_t_data.txt")
output_file = Path("/home/joris/Datadir/ModelingCamp2025/DatasetReduced/reduced_t_data.csv")

df = pd.read_csv(input_file, delim_whitespace=True, header=None, names=["cy_id", "F1", "F2", "F3"])

# Convert the Year column to the required format
df["cy_id"] = df["cy_id"].astype(int).astype(str) + "_1_1"

# Display the first few rows to verify
print(df.tail(20))

df.to_csv(output_file, index=False)
