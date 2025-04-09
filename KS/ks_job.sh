#!/bin/bash

#SBATCH --job-name=train_ks
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=010:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=reusebi@caltech.edu
#SBATCH --reservation=clima

conda activate neuralop

python /central/groups/esm/reusebi/MLStability/KS_FNO/KS/train_ks_time.py --data.coarsen_factor 32 --fno.n_modes [16]  --opt.n_epochs 50 --fno.n_layers 4 --fno.hidden_channels 64 --t_points 1
