from pathlib import Path
import pandas as pd

input_file = Path("../../data/PECDc4.2_NLregion_TAfeatures_Vu.csv")
output_file = Path("../../data/PECDc4.2_NLregion_TAfeatures_Vu_Years.csv")


# Read CSV file into DataFrame
df = pd.read_csv(input_file)

# Extract the 4-digit year from the first column
df.insert(1, 'Year', df.iloc[:, 0].str.extract(r'(\d{4})'))
df['Year'] = df['Year'].astype(float).round(1)

# Display the resulting DataFrame
with pd.option_context('display.max_rows', 100,
                       'display.min_rows', 50,
                       'display.max_columns', None,
                       'display.width', 1000,
                       'display.precision', 8,
                       ):
    print(df)

df.to_csv(output_file, index=False)
