#!/bin/bash

# -----------------------------
# Make executable
# chmod +x eval.sh
# Run it:
# ./eval.sh
# -----------------------------

# -----------------------------
# Initialize conda in this shell
# -----------------------------
# This line works for both Linux and macOS
eval "$(conda shell.bash hook)"

# -----------------------------
# Activate the correct environment
# -----------------------------
conda activate smol311
echo "Using conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"
echo ""

# -----------------------------
# List of workloads
# -----------------------------
datasets=("astronomy" "archeology" "biomedical" "environment" "legal" "wildfire") # "environment" "legal" "wildfire"

sut="SmolagentsPDTClaude37Sonnet"
cmd="python evaluate.py"

echo "Starting evaluations for SUT: $sut"
echo ""

# -----------------------------
# Main loop
# -----------------------------
for ds in "${datasets[@]}"; do
    echo "============================================"
    echo "Running workload: $ds"
    echo "============================================"
    
    $cmd --sut "$sut" --workload "$ds" --use_truth_subset

    echo ""
    echo "Finished workload: $ds"
    echo ""
done

echo "All evaluations completed."