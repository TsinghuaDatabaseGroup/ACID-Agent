#!/usr/bin/env python
# coding: utf-8
import pandas as pd
data_path = "./data/archeology/input/"

roman_path = f"{data_path}/roman_cities.csv"
roman_df = pd.read_csv(roman_path)

filtered_df = roman_df[roman_df["Select Bibliography"].notna()]
roman_sources = filtered_df["Select Bibliography"]

sources = set()
index = 0
for values in roman_sources:
    references = values.split(";")
    for ref in references:
        ref = ref.strip()
        ref = ref.replace("?", "").replace(".", "")
        ref = ref.split(":")[0]
        if ref != "":
            sources.add(ref)

print(len(sources))