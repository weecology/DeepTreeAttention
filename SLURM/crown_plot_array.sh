#!/bin/bash
#SBATCH --job-name=dt_crown_plot
#SBATCH --account=ewhite
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/crown_plot_%A_%a.out
#SBATCH --error=logs/crown_plot_%A_%a.err
##SBATCH --array=1-50%10

# One task per line in plots.txt (plotID). Example:
#   awk 'NR>1{print $1}' plots_export.csv | sort -u > plots.txt   # if first column is plotID
# Then: #SBATCH --array=1-$(wc -l < plots.txt)%10

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$HOME/DeepTreeAttention}"
export CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/config.yml}"
CANOPY_POINTS="${CANOPY_POINTS:-$REPO_ROOT/data/interim/canopy_points.shp}"
PLOTS_FILE="${PLOTS_FILE:-$REPO_ROOT/plots.txt}"

cd "$REPO_ROOT"
mkdir -p logs

LINE_NO="${SLURM_ARRAY_TASK_ID:?Set SLURM_ARRAY_TASK_ID or submit with sbatch --array}"
PLOT="$(sed -n "${LINE_NO}p" "$PLOTS_FILE")"
if [[ -z "${PLOT}" ]]; then
  echo "No plot on line ${LINE_NO} of ${PLOTS_FILE}"
  exit 1
fi

RGB_GLOB="$(uv run python -c "import os; from src import utils; print(utils.read_config(os.environ['CONFIG_PATH'])['rgb_sensor_pool'])")"

uv run python -m src.pipelines.crown_one_plot \
  --canopy-points "$CANOPY_POINTS" \
  --plot "$PLOT" \
  --rgb-glob "$RGB_GLOB" \
  --savedir "${CROWN_BOX_DIR:-$REPO_ROOT/data/interim/boxes}" \
  --raw-box-savedir "${RAW_BOX_DIR:-$REPO_ROOT/data/interim/raw_boxes}"
