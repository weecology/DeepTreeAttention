#!/bin/bash
#SBATCH --job-name=DeepTreeAttention_clean   # Job name
#SBATCH --mail-type=END               # Mail events
#SBATCH --mail-user=benweinstein2010@gmail.com  # Where to send mail
#SBATCH --account=ewhite
#SBATCH --cpus-per-task=10
#SBATCH --mem=40GB
#SBATCH --time=24:00:00       #Time limit hrs:min:sec
#SBATCH --output=/home/b.weinstein/logs/DeepTreeAttention_clean_%j.out   # Standard output and error log
#SBATCH --error=/home/b.weinstein/logs/DeepTreeAttention_clean_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1

module load tensorflow/2.3.1
module load cuda

export PATH=${PATH}:/home/b.weinstein/miniconda3/envs/DeepTreeAttention_38/bin/
export PYTHONPATH=/home/b.weinstein/miniconda3/envs/DeepTreeAttention_38/lib/python3.8/site-packages/:/home/b.weinstein/DeepTreeAttention/:${PYTHONPATH}
export LD_LIBRARY_PATH=/home/b.weinstein/miniconda3/envs/DeepTreeAttention_38/lib/:${LD_LIBRARY_PATH}

#Run using tensorflow greater than tensorflow 2.0
cd /home/b.weinstein/DeepTreeAttention/experiments/Trees/

#python generate.py
#export COMET_LOGGING_FILE=~/tmp/comet.log
#export COMET_LOGGING_CONSOLE=debug

python clean.py
