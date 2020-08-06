#!/bin/bash
#SBATCH --job-name=DeepTreeAttention   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --qos=ewhite-b
#SBATCH --cpus-per-task=10
#SBATCH --mem=30GB
#SBATCH --time=72:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/DeepTreeAttention_%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/DeepTreeAttention_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=2

module load tensorflow

export PATH=${PATH}:/home/b.weinstein/miniconda3/envs/DeepTreeAttention/bin/
export PYTHONPATH=${PYTHONPATH}:/home/b.weinstein/miniconda3/envs/DeepTreeAttention/lib/python3.7/site-packages/
export LD_LIBRARY_PATH=/home/b.weinstein/miniconda3/envs/DeepTreeAttention/lib/:${LD_LIBRARY_PATH}

cd /home/b.weinstein/DeepTreeAttention/experiments/NEON/

#Generate only once
python run.py
