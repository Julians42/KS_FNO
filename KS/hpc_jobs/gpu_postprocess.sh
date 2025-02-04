#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1  # Request specifically an H100 GPU
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --output=out_tunnel.out

module purge

./../../../code tunnel