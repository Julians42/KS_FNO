#!/bin/bash

#SBATCH --time=4:00:00
#SBATCH --mem=64GB        # Increased memory to 64GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # Increased to 4 tasks per node
#SBATCH --cpus-per-task=8   # Allocating 16 CPUs per task (total 64 CPUs)
#SBATCH --partition=expansion
#SBATCH --output=out_tunnel.out


module purge

./../../../code tunnel