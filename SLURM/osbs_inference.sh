#!/bin/bash
#SBATCH --job-name=osbs_infer
#SBATCH --mail-type=END
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --output=/home/b.weinstein/logs/osbs_inference_%j.out
#SBATCH --error=/home/b.weinstein/logs/osbs_inference_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1

# OSBS tile inference (detection + species). Configure inference_osbs in config.yml, then submit.

set -euo pipefail

ulimit -c 0

REPO_ROOT="${REPO_ROOT:-${HOME}/DeepTreeAttention}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config.yml}"

module load git gcc 2>/dev/null || true
source activate DeepTreeAttention

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

python -m src.pipelines.osbs_inference --config "${CONFIG_PATH}"
