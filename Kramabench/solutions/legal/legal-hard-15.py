import pandas as pd
import os

# Load all state-level MSA identity theft files.
data_path = "./data/legal/input/"
directory = f"{data_path}/csn-data-book-2024-csv/CSVs/State MSA Identity Theft data"

state_data = {}
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    state_data[filename.split(".")[0]] = pd.read_csv(filepath, skiprows=2).dropna()

# Combine all state files into one dataframe.
overall_df = pd.concat(state_data.values(), ignore_index=True).reset_index(drop=True)

# Extract the state portion from names like "Area Name, ST".
overall_df["states"] = overall_df["Metropolitan Area"].apply(
    lambda x: x.split(",")[1].split()[0] if isinstance(x, str) and "," in x else None
)

# Cross-state MSAs have state abbreviations connected by a hyphen.
overall_df["is_cross_state"] = overall_df["states"].apply(
    lambda x: isinstance(x, str) and "-" in x
)

# Keep true Metropolitan Statistical Areas, excluding Micropolitan Statistical Areas.
overall_df["is_metropolitan"] = (
    overall_df["Metropolitan Area"].str.contains("Metropolitan Statistical Area", case=False, na=False)
    & ~overall_df["Metropolitan Area"].str.contains("Micropolitan Statistical Area", case=False, na=False)
)

# Convert report counts from comma-formatted strings to integers.
overall_df["# of Reports"] = (
    overall_df["# of Reports"]
    .apply(lambda x: x.replace(",", "") if isinstance(x, str) else x)
    .astype(int)
)

# Remove duplicate rows that may appear across state files.
overall_df.drop_duplicates(inplace=True)

# Sum reports for cross-state Metropolitan Statistical Areas only.
answer = overall_df[
    overall_df["is_cross_state"] & overall_df["is_metropolitan"]
]["# of Reports"].sum()

print(answer)