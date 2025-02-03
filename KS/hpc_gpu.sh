#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

#SBATCH --partition=gpu
#SBATCH --output=out_tunnel.out

module purge

./../../code tunnel