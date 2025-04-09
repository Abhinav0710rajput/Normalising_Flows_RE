#!/bin/bash
#SBATCH --partition=epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=24:00:00

module load cuda/11.1-1
module load cudnn/8.0.4.30-11.1

python main.py --configs config.yaml
