#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --account=def-wan
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --output=%N-%j.out
cd ~/projects/def-wan/seantang/InstaNovo

module purge
module load python/3.11
source .venv/bin/activate

python train_diffusion_marg.py