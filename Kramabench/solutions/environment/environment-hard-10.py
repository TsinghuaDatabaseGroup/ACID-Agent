import pandas as pd
import os
data_path = "./data/environment/input"

def prepare_beach_datasheet(fp:str) -> pd.DataFrame:
    """
    Prepare the beach datasheet for analysis.
    Args:
        fp (str): File path to the beach datasheet CSV file.
    Returns:
        pd.DataFrame: Prepared DataFrame with relevant columns.
    """
    # Step 0: Check if the file exists
    if not os.path.exists(fp):
        raise FileNotFoundError(f"File not found: {fp}")
    # Step 1: Skip first row, read the next two as headers
    df = pd.read_csv(fp, skiprows=1, header=[0, 1])

    # Step 2: Flatten multi-level columns
    cols = df.columns.to_frame()
    cols[0] = cols[0].replace(r'^Unnamed.*', None, regex=True).ffill()
    df.columns = ['_'.join(col).strip() if col[0] is not None else col[1] for col in cols.itertuples(index=False, name=None)]

    # Step 3: Identify ID columns and melt the rest
    location_cols = [col for col in df.columns if 'Tag' in col or 'Enterococcus' in col]
    id_cols = [col for col in df.columns if col not in location_cols]

    melted = df.melt(id_vars=id_cols, value_vars=location_cols, 
                    var_name='Variable', value_name='Value')

    # Step 4: Extract Location and Measure from Variable column
    melted['Location'] = melted['Variable'].apply(lambda x: x.split('_')[0])
    melted['Measure'] = melted['Variable'].apply(lambda x: x.split('_')[1])

    # Step 5: Pivot to tidy format
    df = melted.pivot(index=id_cols + ['Location'], columns='Measure', values='Value').reset_index()

    # Optional: flatten column names
    df.columns.name = None

    # Step 6: cast to numeric
    for col in df.columns:
        if col not in id_cols + ['Location', 'Tag']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

ej_df = pd.read_csv(os.path.join(data_path, 'environmental-justice-populations.csv'))
more_than_90 = ej_df[ej_df['Percent of population in EJ BGs'] > 90]
more_than_90_cities = more_than_90['Municipality'].unique()
# print(f"Cities with more than 90% of the population in EJ BGs: {len(more_than_90_cities)}")

df = pd.read_csv(os.path.join(data_path, 'water-body-testing-2023.csv'))
beach_type = "Marine"
df = df[df['Beach Type Description'] == beach_type]
# Use more_than_90_cities to filter the dataframe ['County']
df1 = df[df['Community'].isin(more_than_90_cities)]
# Split the 'Beach Name' column by '@' and keep only the first part, since the second part is the location of where the sample was taken
df1["Beach Name"] = df1['Beach Name'].str.split('@').str[0]
df1['Beach Name'].unique()

boston_beach_names = ['constitution_beach_datasheet.csv', 'pleasure_bay_and_castle_island_beach_datasheet.csv', 
                    'city_point_beach_datasheet.csv', 'm_street_beach_datasheet.csv', 'carson_beach_datasheet.csv', 
                    'malibu_beach_datasheet.csv', 'tenean_beach_datasheet.csv', 'wollaston_beach_datasheet.csv']
# Get the beach names from boston_beach_names without the '_datasheet.csv' suffix and with spaces instead of underscores, and title case
formatted_boston_beach_names = [name.replace('_beach_datasheet.csv', '') for name in boston_beach_names]
# If and in the beach name, split by 'and' and keep both parts
final_boston_beach_names = []
for name in formatted_boston_beach_names:
    if 'and' in name:
        parts = name.split('_and_')
        for part in parts:
            final_boston_beach_names.append(part.replace('_', ' ').title())
    else:
        final_boston_beach_names.append(name.replace('_', ' ').title())
print(f"Boston beach names: {final_boston_beach_names}")

# Take a union of the beach names in df1 and final_boston_beach_names
df1_beach_names = df1['Beach Name'].str.title().unique().tolist()
# strip whitespace from beach names in df1_beach_names
df1_beach_names = [name.strip() for name in df1_beach_names]
common_beaches = set(df1_beach_names).intersection(set(final_boston_beach_names))
print(f"Common beaches between EJ >90% communities and Boston beaches: {common_beaches}")

# Find the beach datasheet for common_beaches
target_beach_files = []
for beach in common_beaches:
    beach_file_name = beach.lower()
    for f in boston_beach_names:
        if beach_file_name in f:
            target_beach_files.append((beach, f))
            break

print(f"Target beach files: {target_beach_files}")

dfs = []
# For each target beach file, get the dataframe, and union them
for beach, beach_file in target_beach_files:
    df = prepare_beach_datasheet(os.path.join(data_path, beach_file))
    # Impute missing values in '1-Day Rain', '2-Day Rain', '3-Day Rain', 'Enterococcus'
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# Compute the correlation between 'Total Rain' and 'Enterococcus'
correlation = df['3-Day Rain'].corr(df['Enterococcus'], method='pearson')
print(f"Correlation between Total Rain and Enterococcus: {correlation:.3f}")