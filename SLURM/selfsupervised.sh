#!/bin/bash
#SBATCH --job-name=selfsupervised   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --nodes=1                 # Number of MPI ran
#SBATCH --cpus-per-task=1
#SBATCH --mem=50GB
#SBATCH --time=48:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/DeepTreeAttention_%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/DeepTreeAttention_%j.err

source activate DeepTreeAttention

cd ~/DeepTreeAttention/
python notebooks/crop_random_tile.py