#!/bin/bash

module load anaconda/Python-ML-2025a
source .venv/bin/activate

# Run the script
datasets=("environment" "wildfire" "astronomy" "biomedical" "archeology" "legal")

for ds in "${datasets[@]}"; do
  echo "Running benchmark on dataset: $ds"
  python evaluate.py --dataset_name "$ds" --workload_filename "$ds.json"
done