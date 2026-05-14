# DeepTreeAttention

[Github Actions](https://github.com/Weecology/DeepTreeAttention/actions/)

Tree species classification for **National Ecological Observatory Network (NEON)** imagery, implementing Hang et al. 2020 ([Hyperspectral Image Classification with Attention Aided CNNs](https://arxiv.org/abs/2005.11977)) with a PyTorch Lightning training stack.

This README is organized around the lifecycle you described: **raw points → generated tensors → training → evaluation → inference → reporting**. Dask has been removed in favor of **single-node sequential code** plus **plain SLURM job arrays** for embarrassingly parallel stages (per-plot crowns, large I/O batches, and so on).

---

## 0. Environment (uv)

```bash
uv sync --extra dev
uv run pytest -v
```

Use `uv run …` for every CLI invocation so the locked environment is respected.

---

## 1. Data layout and the “HPC handoff”


| Stage                                     | Location                                         | What it is                                                                                         |
| ----------------------------------------- | ------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| **Upstream (not in git)**                 | e.g. `/orange/ewhite/NeonData/...` on HiPerGator | Full NEON mirror; too large to vendor.                                                             |
| **Shareable inputs**                      | `data/raw/`                                      | VST (or similar) stem tables, AOI shapefiles. See `data/raw/README.md`.                            |
| **Intermediary bundle (commit or rsync)** | `data/interim/`                                  | Filtered stem points (`canopy_points.shp`), optional per-plot crown boxes before merge, manifests. |
| **Heavy rasters**                         | `data/external/neon_aop/`                        | RGB / HSI / CHM tiles from `**neonutilities`** downloads or selective `rsync` from the mirror.     |
| **Training tensors**                      | `data/processed/<run>/` or `config["data_dir"]`  | HSI crops, `train.csv` / `test.csv`, `crowns.shp`.                                                 |


**Practical handoff from HiPerGator:** export the VST CSV you trust, copy it to `data/raw/`, then either (a) run `deeptree-download-aop` (below) into `data/external/neon_aop/`, or (b) `rsync` only the DP3 products you need into the same tree. Point `rgb_sensor_pool`, `HSI_sensor_pool`, and `CHM_pool` in `config.yml` at that mirror. You no longer need to regenerate crops for every experiment if you set `use_data_commit` to a frozen directory name or reuse `data_dir` with `replace: false` after the first successful build.

---

## 2. Download AOP tiles (`neonutilities`)

The Python `**neonutilities`** package exposes `by_tile_aop`, equivalent to the R `neonUtilities::byTileAOP` helper. This repository wraps it for stem coordinates.

1. Obtain a NEON API token and export it (optional for tiny pulls; recommended otherwise):
  ```bash
   export NEON_API_TOKEN="your_token"
  ```
2. After you have `canopy_points.shp` (from the filtering stage in `TreeData`), set `data_layout.canopy_points_shp` in `config.yml` **or** pass `--points`.
3. Run:
  ```bash
   uv run deeptree-download-aop --config config.yml
  ```
   Defaults live under `neon_download` in `config.yml` (site, years, DP3 IDs for RGB, hyperspectral, CHM, buffer in meters).

---

## 3. Generate datasets (0 — data generation)

Pipeline code paths:

- `src/data.py` — filter VST, CHM checks, megaplot merge hooks, train/test split.
- `src/generate.py` — DeepForest crowns, hyperspectral crops, optional H5→TIF conversion via `src/neon_paths.py`.

**Local / single job:** instantiate `data.TreeData` with `use_data_commit: null` and your `config.yml`, as in `train.py`.

**Parallel crowns (SLURM):** each array task runs one plot:

```bash
uv run python -m src.pipelines.crown_one_plot \
  --canopy-points data/interim/canopy_points.shp \
  --plot OSBS_001 \
  --rgb-glob "data/external/neon_aop/**/DP3.30010.001/**/Camera/**/*.tif" \
  --savedir data/interim/boxes \
  --raw-box-savedir data/interim/raw_boxes
```

After all tasks finish, merge:

```bash
uv run deeptree-merge-crown-boxes \
  --boxes-dir data/interim/boxes \
  --out data/interim/crowns.shp
```

Submit `SLURM/crown_plot_array.sh` (tune `#SBATCH` directives, `plots.txt`, and `REPO_ROOT`).

> **Note:** Passing a Dask `Client` into `points_to_crowns`, `generate_crops`, or `train_test_split` now raises a clear error. Parallelize with SLURM (or your own outer loop), not an in-Python Dask cluster.

---

## 4. Training (1)

**Experiment identity (Comet + SLURM)** — you no longer pass git branch/sha as required positionals. The trainer auto-detects git (branch, SHA, dirty flag) and uploads a merged `**config.merged.yml`**, optional `**git_diff_uncommitted.patch**`, and a `**src/**` code bundle to Comet when logging is enabled.

```bash
# Human-readable Comet experiment name (recommended)
export DEEPTREE_EXPERIMENT_NAME=osbs-baseline-epoch70
uv run python train.py --config config.yml --experiment-name "${DEEPTREE_EXPERIMENT_NAME}"

# Or rely on auto naming (UTC timestamp + short SHA, or SLURM_JOB_ID on clusters)
uv run python train.py --config config.yml

# Optional YAML merged last (keep one file per ablation under e.g. experiments/)
uv run python train.py --config config.yml --overrides experiments/my_ablation.yml -n osbs-lr-sweep-001
```

Environment variables (highest priority first for the display name): `**DEEPTREE_EXPERIMENT_NAME**`, `**COMET_EXPERIMENT_NAME**`, then `**--experiment-name**`, then a default from `**SLURM_JOB_ID**` or timestamp.

`train.py` reads `config.yml` (+ `config.local.yml` if present), logs to Comet when `**use_comet: true**` and `**COMET_API_KEY**` / `**COMET_KEY**` are set, and writes checkpoints under `**checkpoint_dir**`. `raw_vst_csv` defaults to `data/raw/neon_vst_data_2022.csv` but can be overridden in YAML.

**SLURM (queued GPU training)** — from the repo checkout, set **`EXPERIMENT_NAME`** (forwarded as **`DEEPTREE_EXPERIMENT_NAME`**), optionally **`DEEPTREE_OVERRIDES`** and **`DEEPTREE_CONFIG`**, then:

```bash
cd /path/to/DeepTreeAttention
EXPERIMENT_NAME=osbs-baseline sbatch SLURM/train_experiment.sh
```

**`REPO_ROOT`** defaults to **`SLURM_SUBMIT_DIR`** (your shell’s current directory when you run `sbatch`), so you normally **do not** export it if you `cd` into the project first. Set **`REPO_ROOT=/path/to/DeepTreeAttention`** only when you submit from another directory.

Edit `#SBATCH` lines in `SLURM/train_experiment.sh` for your partition/account. The job runs from a **shared checkout** so you can **`git pull`** or edit **`experiments/*.yml`** between submissions; each job still logs the **resolved config** and **git SHA** to Comet for apples-to-apples comparison.

---

## 5. Evaluation (2)

Evaluation is integrated in the Lightning `TreeModel` / `MultiStage` path (validation metrics, crown-level scoring). Point `use_data_commit` at a frozen processed directory to re-score without touching generation.

---

## 6. Inference (3)

**OSBS RGB tile inference** is configured solely through `inference_osbs` in `config.yml`.

```bash
uv run python predict.py --config config.yml
# or
uv run deeptree-infer-osbs --config config.yml
```

HiPerGator template: `SLURM/osbs_inference.sh`.

**OSBS mortality comparison** is configured through `osbs_mortality` in `config.yml`. It downloads AOI-intersecting RGB tiles when requested, runs the detector plus alive/dead cropmodel, optionally runs species inference, and writes tile-level CSV/GPKG outputs plus GeoTIFF rasters for dead counts and dead-count change.

```bash
uv run deeptree-osbs-mortality --config config.yml
```

HiPerGator template: `SLURM/osbs_mortality.sh`.

---

## 7. Reporting and analysis (4)

- Comet dashboards (when `comet_ml` is configured).
- Scripts such as `abundance.py`, `create_prediction_shp.py`, and `src/multinomial.py` now use `ThreadPoolExecutor` instead of Dask for light parallelism over shapefiles.

---

## Project map

```text
├── config.yml              # Central configuration (+ neon_download / data_layout)
├── data/
│   ├── raw/README.md       # What belongs in raw inputs + rsync hints
│   ├── external/           # Downloaded / rsync'd NEON tiles (.gitkeep only)
│   └── interim/            # Optional canonical intermediate artifacts
├── experiments/            # Optional YAML fragments for ``--overrides`` (one ablation per file)
├── SLURM/                  # Job scripts (``train_experiment.sh``, inference, crown array)
├── src/
│   ├── data.py             # Lightning TreeData + filtering
│   ├── experiment_tracking.py  # Git metadata + default experiment names
│   ├── generate.py         # Crowns + crops
│   ├── neon_download.py    # neonutilities helpers
│   └── pipelines/          # CLIs (inference, download, crown worker, merge)
├── train.py                # Full training driver
├── predict.py              # OSBS inference entry
└── pyproject.toml          # Dependencies + `deeptree-*` console scripts
```

---

## Open questions / follow-up work

1. **Megaplot + IFAS branches** — logic is dense; consider isolating into a small submodule with explicit tests.
2. **CHM product ID per NEON revision** — confirm `DP3.30015.001` matches your mirror layout (`CanopyHeightModelGtif`).
3. **Comet as the sole reproducibility anchor** — `use_data_commit` ties runs to Comet artifact IDs; consider replacing with explicit semantic version tags on `data/processed/<name>/`.
4. **Dead-tree filtering in `TreeData`** — references `self.predicted_dead` in a logging block; verify that attribute is always defined on your code path before relying on those images in Comet.

Training uses Lightning 2 `Trainer(accelerator=…, devices=…)` (see `devices` / `accelerator` in `config.yml`; legacy `gpus` is still read as a device count when `devices` is omitted).

Contributions: branch per feature, add/adjust pytest coverage for anything you touch, keep diffs focused.