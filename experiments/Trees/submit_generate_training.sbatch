#!/bin/bash
#SBATCH --job-name=generate   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --cpus-per-task=10
#SBATCH --mem=30GB
#SBATCH --time=72:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/GenerateTreeAttention_%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/GenerateTreeAttention_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1

module load tensorflow/1.14.0

export PATH=${PATH}:/home/b.weinstein/miniconda3/envs/DeepTreeAttention_DeepForest/bin/:/home/b.weinstein/DeepTreeAttention/
export PYTHONPATH=/home/b.weinstein/miniconda3/envs/DeepTreeAttention_DeepForest/lib/python3.7/site-packages/:/home/b.weinstein/DeepTreeAttention/:${PYTHONPATH}
export LD_LIBRARY_PATH=/home/b.weinstein/miniconda3/envs/DeepTreeAttention_DeepForest/lib/:${LD_LIBRARY_PATH}

cd /home/b.weinstein/DeepTreeAttention/experiments/Trees/

#Generate only once using DeepForest requires less than tensorflow 2.0
python prepare_field_data.py