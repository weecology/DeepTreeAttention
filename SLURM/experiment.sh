#!/bin/bash
# Same runner as train_experiment.sh with alternate #SBATCH defaults (mail, log paths).
# Submit from the repo (cd .../DeepTreeAttention) or set REPO_ROOT. Optional: EXPERIMENT_NAME, DEEPTREE_OVERRIDES, DEEPTREE_CONFIG.
#SBATCH --job-name=DeepTreeAttention
#SBATCH --mail-type=END
#SBATCH --mail-user=benweinstein2010@gmail.com
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=50GB
#SBATCH --time=48:00:00
#SBATCH --output=/home/b.weinstein/logs/DeepTreeAttention_%j.out
#SBATCH --error=/home/b.weinstein/logs/DeepTreeAttention_%j.err
#SBATCH --partition=gpu
#SBATCH --gpus=1

set -euo pipefail

ulimit -c 0
module load git gcc

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${REPO_ROOT}" ]]; then
  echo "[experiment] Set REPO_ROOT or cd into DeepTreeAttention before sbatch." >&2
  exit 1
fi
if [[ ! -f "${REPO_ROOT}/train.py" ]]; then
  echo "[experiment] REPO_ROOT=${REPO_ROOT} is not the repo root (no train.py)." >&2
  exit 1
fi
cd "${REPO_ROOT}"

mkdir -p logs

export DEEPTREE_EXPERIMENT_NAME="${EXPERIMENT_NAME:-DeepTreeAttention-${SLURM_JOB_ID}}"

OVERRIDES_ARGS=()
if [[ -n "${DEEPTREE_OVERRIDES:-}" ]]; then
  OVERRIDES_ARGS=(--overrides "${DEEPTREE_OVERRIDES}")
fi

CONFIG_PATH="${DEEPTREE_CONFIG:-config.yml}"

if command -v uv >/dev/null 2>&1; then
  RUN=(uv run python train.py --config "${CONFIG_PATH}" "${OVERRIDES_ARGS[@]}" --experiment-name "${DEEPTREE_EXPERIMENT_NAME}")
else
  RUN=(python train.py --config "${CONFIG_PATH}" "${OVERRIDES_ARGS[@]}" --experiment-name "${DEEPTREE_EXPERIMENT_NAME}")
fi

echo "[experiment] SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "[experiment] DEEPTREE_EXPERIMENT_NAME=${DEEPTREE_EXPERIMENT_NAME}"
echo "[experiment] REPO_ROOT=${REPO_ROOT}"
echo "[experiment] running: ${RUN[*]}"

exec "${RUN[@]}"
