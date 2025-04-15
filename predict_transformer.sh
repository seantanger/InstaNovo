#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=def-wan
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --output=%N-%j.out
cd ~/projects/def-wan/seantang/InstaNovo

module purge
module load python/3.11
source .venv/bin/activate

instanovo transformer predict \
    --instanovo-model /home/seantang/projects/def-wan/seantang/InstaNovo/checkpoints/instanovo-v1.1.0.ckpt \
    --data-path /home/seantang/projects/def-wan/seantang/InstaNovo/data/ms_proteometools/test.ipc \
    --output-path instanovo_transformer_predictions_acpt.csv \
    --evaluation \