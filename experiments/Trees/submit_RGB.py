#!/bin/bash
#SBATCH --job-name=DeepTreeAttention   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --cpus-per-task=10
#SBATCH --mem=30GB
#SBATCH --time=72:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/DeepTreeAttention_%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/DeepTreeAttention_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=4

module load tensorflow/2.1.0

export PATH=${PATH}:/home/b.weinstein/miniconda3/envs/DeepTreeAttention/bin/
export PYTHONPATH=/home/b.weinstein/miniconda3/envs/DeepTreeAttention/lib/python3.7/site-packages/:/home/b.weinstein/DeepTreeAttention/:${PYTHONPATH}
export LD_LIBRARY_PATH=/home/b.weinstein/miniconda3/envs/DeepTreeAttention/lib/:${LD_LIBRARY_PATH}

#Run using tensorflow greater than tensorflow 2.0
cd /home/b.weinstein/DeepTreeAttention/experiments/Trees/

python run_RGB.py
