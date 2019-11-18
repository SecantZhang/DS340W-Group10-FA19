#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM-shared
#SBATCH -t 5:00:00
#SBATCH --ntasks-per-node 1

set -e

module load python3
