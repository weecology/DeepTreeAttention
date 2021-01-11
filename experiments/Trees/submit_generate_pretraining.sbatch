#!/bin/bash
#SBATCH --job-name=generate   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --cpus-per-task=1
#SBATCH --mem=40GB
#SBATCH --time=24:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/GenerateTreeAttention_%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/GenerateTreeAttention_%j.err

module load tensorflow/2.3.1

export PATH=${PATH}:/home/b.weinstein/miniconda3/envs/DeepTreeAttention_38/bin/:/home/b.weinstein/DeepTreeAttention/
export PYTHONPATH=/home/b.weinstein/miniconda3/envs/DeepTreeAttention_38/lib/python3.8/site-packages/:/home/b.weinstein/DeepTreeAttention/:${PYTHONPATH}
export LD_LIBRARY_PATH=/home/b.weinstein/miniconda3/envs/DeepTreeAttention_38/lib/:${LD_LIBRARY_PATH}

cd /home/b.weinstein/DeepTreeAttention/experiments/Trees/

#Generate only once using DeepForest requires less than tensorflow 2.0
python generate_unsupervised.py