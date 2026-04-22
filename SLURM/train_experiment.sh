#!/bin/bash
# Submit a queued GPU training run with a stable experiment identity for Comet.
#
# Usage (from login node, repo checked out on shared FS):
#   export REPO_ROOT=/path/to/DeepTreeAttention
#   export EXPERIMENT_NAME=osbs-epoch70-bs128-$(date +%Y%m%d)
#   # optional: point at a YAML fragment merged last over config.yml
#   export DEEPTREE_OVERRIDES="$REPO_ROOT/experiments/osbs_ablation.yml"
#   sbatch SLURM/train_experiment.sh
#
# Comet: set COMET_API_KEY (and optionally COMET_WORKSPACE) in the environment
# or load them from a secrets file before sbatch. DEEPTREE_EXPERIMENT_NAME is
# forwarded so the Comet UI name matches your SLURM intent.

#SBATCH --job-name=dt-train
#SBATCH --account=ewhite
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --output=logs/train_%x_%j.out
#SBATCH --error=logs/train_%x_%j.err

set -euo pipefail

: "${REPO_ROOT:?Set REPO_ROOT to the DeepTreeAttention checkout on this cluster}"
cd "${REPO_ROOT}"

mkdir -p logs

# Stable name for Comet + log files (override when submitting)
export DEEPTREE_EXPERIMENT_NAME="${EXPERIMENT_NAME:-train-${SLURM_JOB_ID}}"

# Optional merged config fragment (same as train.py --overrides)
OVERRIDES_ARGS=()
if [[ -n "${DEEPTREE_OVERRIDES:-}" ]]; then
  OVERRIDES_ARGS=(--overrides "${DEEPTREE_OVERRIDES}")
fi

CONFIG_PATH="${DEEPTREE_CONFIG:-config.yml}"

# Use uv if available (recommended); fall back to python on the module path.
if command -v uv >/dev/null 2>&1; then
  RUN=(uv run python train.py --config "${CONFIG_PATH}" "${OVERRIDES_ARGS[@]}" --experiment-name "${DEEPTREE_EXPERIMENT_NAME}")
else
  RUN=(python train.py --config "${CONFIG_PATH}" "${OVERRIDES_ARGS[@]}" --experiment-name "${DEEPTREE_EXPERIMENT_NAME}")
fi

echo "[train_experiment] SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "[train_experiment] DEEPTREE_EXPERIMENT_NAME=${DEEPTREE_EXPERIMENT_NAME}"
echo "[train_experiment] REPO_ROOT=${REPO_ROOT}"
echo "[train_experiment] running: ${RUN[*]}"

exec "${RUN[@]}"
