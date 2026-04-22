# DeepTreeAttention

[![Github Actions](https://github.com/Weecology/DeepTreeAttention/actions/workflows/pytest.yml/badge.svg)](https://github.com/Weecology/DeepTreeAttention/actions/)

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

| Stage | Location | What it is |
|-------|-----------|------------|
| **Upstream (not in git)** | e.g. `/orange/ewhite/NeonData/...` on HiPerGator | Full NEON mirror; too large to vendor. |
| **Shareable inputs** | `data/raw/` | VST (or similar) stem tables, AOI shapefiles. See `data/raw/README.md`. |
| **Intermediary bundle (commit or rsync)** | `data/interim/` | Filtered stem points (`canopy_points.shp`), optional per-plot crown boxes before merge, manifests. |
| **Heavy rasters** | `data/external/neon_aop/` | RGB / HSI / CHM tiles from **`neonutilities`** downloads or selective `rsync` from the mirror. |
| **Training tensors** | `data/processed/<run>/` or `config["data_dir"]` | HSI crops, `train.csv` / `test.csv`, `crowns.shp`. |

**Practical handoff from HiPerGator:** export the VST CSV you trust, copy it to `data/raw/`, then either (a) run `deeptree-download-aop` (below) into `data/external/neon_aop/`, or (b) `rsync` only the DP3 products you need into the same tree. Point `rgb_sensor_pool`, `HSI_sensor_pool`, and `CHM_pool` in `config.yml` at that mirror. You no longer need to regenerate crops for every experiment if you set `use_data_commit` to a frozen directory name or reuse `data_dir` with `replace: false` after the first successful build.

---

## 2. Download AOP tiles (`neonutilities`)

The Python **`neonutilities`** package exposes `by_tile_aop`, equivalent to the R `neonUtilities::byTileAOP` helper. This repository wraps it for stem coordinates.

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

```bash
uv run python train.py "$(git branch --show-current)" "$(git rev-parse HEAD)"
```

`train.py` reads `config.yml`, logs to Comet when configured, and writes checkpoints to the path in your config. `raw_vst_csv` defaults to `data/raw/neon_vst_data_2022.csv` but can be overridden in YAML.

GPU SLURM example: `SLURM/experiment.sh` (update `conda`/`uv` usage for your module stack).

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
├── SLURM/                  # Job scripts (GPU train, OSBS inference, crown array)
├── src/
│   ├── data.py             # Lightning TreeData + filtering
│   ├── generate.py         # Crowns + crops
│   ├── neon_download.py    # neonutilities helpers
│   └── pipelines/          # CLIs (inference, download, crown worker, merge)
├── train.py                # Full training driver
├── predict.py              # OSBS inference entry
└── pyproject.toml          # Dependencies + `deeptree-*` console scripts
```

---

## Open questions / follow-up work

1. **Lightning 2.x / torchmetrics API drift** — `TreeModel` still targets the older `Trainer(gpus=…)` style; a dedicated upgrade pass would modernize metrics logging and devices.
2. **Megaplot + IFAS branches** — logic is dense; consider isolating into a small submodule with explicit tests.
3. **CHM product ID per NEON revision** — confirm `DP3.30015.001` matches your mirror layout (`CanopyHeightModelGtif`).
4. **Comet as the sole reproducibility anchor** — `use_data_commit` ties runs to Comet artifact IDs; consider replacing with explicit semantic version tags on `data/processed/<name>/`.
5. **Dead-tree filtering in `TreeData`** — references `self.predicted_dead` in a logging block; verify that attribute is always defined on your code path before relying on those images in Comet.

Contributions: branch per feature, add/adjust pytest coverage for anything you touch, keep diffs focused.
