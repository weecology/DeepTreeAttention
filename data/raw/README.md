# Raw and canonical field data

This folder holds **small, shareable** inputs that define *what* to model. Large NEON AOP imagery should not live in git; use `data/external/` (see repository README) or your HPC mirror.

## Recommended layout

| Path | Description |
|------|-------------|
| `neon_vst.csv` | NEON Vegetation Structure (`vst`) table export (or subset) used as the stem/point source. Configure `raw_vst_csv` in `config.yml`. |
| `OSBSBoundary/` | Optional shapefiles for inference AOI (see `inference_osbs.aoi_path` in `config.yml`). |

## Intermediary step between HPC dump and this repo

On HiPerGator (or another host) you typically have a full NEON mirror under paths like `/orange/ewhite/NeonData/...`. **Treat that as upstream storage.** The reproducible hand-off into this repository is:

1. **Export the VST (or megaplot) table** you actually trained on (CSV is enough for `filter_data()` in `src/data.py`).
2. **Option A — download AOP locally** with `neonutilities` (see top-level README, `deeptree-download-aop`) into `data/external/neon_aop/`, then point `rgb_sensor_pool`, `HSI_sensor_pool`, and `CHM_pool` at globs under that tree.
3. **Option B — rsync only the tiles you need** from the HPC mirror into `data/external/neon_aop/` (or another path), using the same basename/geoindex conventions NEON uses so existing `neon_paths.find_sensor_path` logic still works.

After that, set `sensor_download_root` in `config.yml` (or override the `*_pool` globs) and you can run training without touching the full 20+ TB archive.

## Updating woody vegetation structure (VST)

You can refresh the canonical VST table and compare against the 2022 baseline with:

```bash
export NEON_API_TOKEN=...   # recommended to avoid API rate limits
deeptree-update-vst \
  --baseline-csv data/raw/neon_vst_data_2022.csv \
  --output-csv data/raw/neon_vst_data_latest.csv \
  --summary-csv results/vst_additions_by_site_since_2022.csv \
  --new-ids-csv results/vst_new_individual_ids_not_in_2022.csv
```

Notes:

- By default, this updater includes provisional releases and requests records from `2022-01` onward.
- The summary csv reports total rows and unique `individualID` deltas per site.
- Some sites may fail due transient upstream package issues; failed site codes are printed in the JSON summary.

## Example rsync (replace user, host, and remote paths)

```bash
mkdir -p data/external/neon_aop
rsync -avz --progress \
  USER@hpg.rc.ufl.edu:/orange/ewhite/NeonData/OSBS/DP3.30010.001/ \
  data/external/neon_aop/OSBS/DP3.30010.001/
```

Repeat for `DP3.30006.001` (hyperspectral) and your CHM product path as needed.
