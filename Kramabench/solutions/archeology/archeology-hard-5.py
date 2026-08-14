#!/usr/bin/env python
# coding: utf-8

import pandas as pd

# Load the radiocarbon and climate measurement datasets.
data_path = "./data/archeology/input/"

radio = pd.read_excel(data_path + "radiocarbon_database_regional.xlsx")
climate = pd.read_excel(
    data_path + "climateMeasurements.xlsx",
    header=0,
    skiprows=5,
)

# Remove empty rows and columns from both datasets.
radio = radio.dropna(how="all").dropna(axis=1, how="all")
climate = climate.dropna(how="all").dropna(axis=1, how="all")

# Restrict to Neolithic samples from Malta.
malta_neolithic = radio[
    (radio["Region"] == "Malta")
    & (radio["Culture"] == "Neolithic")
].copy()

# Find the northernmost Neolithic Malta sample.
max_lat = malta_neolithic["Latitude"].max()
northern = malta_neolithic[malta_neolithic["Latitude"] == max_lat].copy()

# Dates are BP-style: smaller BP means chronologically later.
target_bp = northern["date"].min()

# Convert climate age from ky to BP-like years, and ensure Al is numeric.
climate["climate_bp"] = pd.to_numeric(climate["Age_ky.1"], errors="coerce") * 1000
climate["Al"] = pd.to_numeric(climate["Al"], errors="coerce")

# Find the climate record closest in time to the selected radiocarbon sample.
dist = (climate["climate_bp"] - target_bp).abs()
closest = climate[dist == dist.min()]

# If multiple rows are equally close, use the maximum Aluminum value.
answer = closest["Al"].max()
print(round(answer, 4))