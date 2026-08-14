import argparse
import csv
import json
import os

import numpy as np
import pandas as pd

from benchmark import Benchmark

workload_names = [
    "archeology.json",
    "astronomy.json",
    "biomedical.json" "environment.json",
    "legal.json",
    "wildfire.json",
]
sut_name = "BaselineLLMSystemGPT4oFewShot"
aggregated_result_filepath = "./results/aggregated_results.csv"

def metric_wise_aggregation():
    aggregated_results_df = pd.read_csv(aggregated_result_filepath)
    metric_aggregation_dict = {}
    for (sut, metric), group in aggregated_results_df.groupby(["sut", "metric"]):
        if sut != sut_name:
            continue
        group_dropped_na = group.dropna()
        metric_aggregation_dict[metric] = group["value_mean"].mean()
    print(metric_aggregation_dict)

metric_wise_aggregation()