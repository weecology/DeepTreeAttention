#!/bin/bash
#SBATCH --job-name=osbs_mortality
#SBATCH --mail-type=END
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --time=72:00:00
#SBATCH --output=/home/b.weinstein/logs/osbs_mortality_%j.out
#SBATCH --error=/home/b.weinstein/logs/osbs_mortality_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1

# OSBS tile-scale mortality comparison. Configure osbs_mortality in config.yml, then submit.

set -euo pipefail

ulimit -c 0

REPO_ROOT="${REPO_ROOT:-${HOME}/DeepTreeAttention}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config.yml}"

module load git gcc 2>/dev/null || true
source activate DeepTreeAttention

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

python -m src.pipelines.osbs_mortality --config "${CONFIG_PATH}"
