#!/bin/bash

# Set script paths (update if needed)
TRAIN_SCRIPT="train_ks.py"
EVAL_SCRIPT="../postprocessing/evaluate_ks.py"

# Define parameter ranges
t_points_list=(1 2 5 10)
coarsen_factors=(1 2 4 8)

# Loop over combinations
for t_points in "${t_points_list[@]}"; do
    for cf in "${coarsen_factors[@]}"; do
        echo "Running with t_points=${t_points}, coarsen_factor=${cf}"

        python "$TRAIN_SCRIPT" --t_points=$t_points --data.coarsen_factor=$cf

        python "$EVAL_SCRIPT" --t_points=$t_points --data.coarsen_factor=$cf
    done
done


# python train_ks.py --t_points=2 --data.coarsen_factor=2
# python ../postprocessing/evaluate_ks.py --t_points=2 --data.coarsen_factor=2