# Train
from __future__ import annotations

import argparse
import os
import sys

import comet_ml
from dotenv import load_dotenv
import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from pandas.util import hash_pandas_object
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CometLogger

from src import data, utils
from src import experiment_tracking
from src.models import multi_stage
from src.local_smoke_logger import LocalSmokeLogger


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DeepTreeAttention training driver",
        epilog="Experiment identity defaults to DEEPTREE_EXPERIMENT_NAME, COMET_EXPERIMENT_NAME, "
        "or SLURM_JOB_ID; git metadata is detected automatically.",
    )
    p.add_argument(
        "--config",
        default="config.yml",
        help="Base config YAML (merged with config.local.yml in the same directory if present)",
    )
    p.add_argument(
        "--overrides",
        default=None,
        help="Optional YAML file merged last (e.g. config.smoke.example.yml for capped batches)",
    )
    p.add_argument(
        "--experiment-name",
        "-n",
        default=None,
        help="Comet experiment display name (also set DEEPTREE_EXPERIMENT_NAME for SLURM)",
    )
    p.add_argument(
        "--git-branch",
        default=None,
        help="Override auto-detected git branch for logging (rare)",
    )
    p.add_argument(
        "--git-sha",
        default=None,
        help="Override auto-detected git SHA for logging (rare)",
    )
    return p


def _make_logger(config: dict, *, experiment_name: str):
    load_dotenv()
    use_comet = config.get("use_comet", True)
    has_key = bool(os.getenv("COMET_API_KEY") or os.getenv("COMET_KEY"))
    if use_comet and has_key:
        return CometLogger(
            project="DeepTreeAttention2",
            workspace=config["comet_workspace"],
            auto_output_logging="simple",
            name=experiment_name,
        )
    return LocalSmokeLogger()


def main_train() -> None:
    args = _build_parser().parse_args()

    config_path = os.path.abspath(args.config)
    repo_root = os.path.dirname(config_path)
    git_root = (
        repo_root if repo_root and os.path.isdir(os.path.join(repo_root, ".git")) else os.getcwd()
    )
    git_meta = experiment_tracking.git_metadata(git_root)
    if args.git_branch:
        git_meta["git_branch"] = args.git_branch
    if args.git_sha:
        git_meta["git_sha"] = args.git_sha
        git_meta["git_short_sha"] = args.git_sha[:7] if len(args.git_sha) > 7 else args.git_sha

    experiment_name = experiment_tracking.comet_display_name(args.experiment_name, git_meta)
    print("[train] experiment_name={}".format(experiment_name), flush=True)

    config = utils.read_config(config_path)
    if args.overrides:
        with open(os.path.abspath(args.overrides), "r") as f:
            config = utils.deep_merge(config, yaml.load(f, Loader=yaml.FullLoader) or {})

    comet_logger = _make_logger(config, experiment_name=experiment_name)

    if config["use_data_commit"]:
        config["crop_dir"] = os.path.join(config["data_dir"], config["use_data_commit"])
    else:
        crop_dir = os.path.join(config["data_dir"], comet_logger.experiment.get_key())
        os.makedirs(crop_dir, exist_ok=True)
        config["crop_dir"] = crop_dir

    # macOS + fork/spawn: DataLoader workers>0 often stalls indefinitely before the first step.
    if sys.platform == "darwin":
        w = int(config.get("workers") or 0)
        if w > 0:
            print(
                "[train] macOS: forcing DataLoader workers=0 (was {}). "
                "num_workers>0 commonly hangs here; set workers: 0 in config.yml to silence."
                .format(w),
                flush=True,
            )
            config["workers"] = 0

    client = None

    comet_logger.experiment.log_parameter("experiment_name", experiment_name)
    comet_logger.experiment.log_parameter("git_branch", git_meta["git_branch"])
    comet_logger.experiment.add_tag(git_meta["git_branch"])
    comet_logger.experiment.log_parameter("git_sha", git_meta["git_sha"])
    comet_logger.experiment.log_parameter("git_short_sha", git_meta["git_short_sha"])
    comet_logger.experiment.log_parameter("git_dirty", bool(git_meta["git_dirty"]))
    comet_logger.experiment.log_parameters(config)

    if isinstance(comet_logger, CometLogger):
        exp = comet_logger.experiment
        try:
            exp.log_asset_data(
                yaml.safe_dump(config, sort_keys=False, default_flow_style=False),
                file_name="config.merged.yml",
            )
        except Exception as exc:
            print("[train] warning: could not log config asset to Comet: {}".format(exc), flush=True)
        if git_meta.get("git_diff_head"):
            try:
                exp.log_asset_data(
                    git_meta["git_diff_head"],
                    file_name="git_diff_uncommitted.patch",
                )
            except Exception as exc:
                print("[train] warning: could not log git diff asset: {}".format(exc), flush=True)
        try:
            exp.log_code(folder=os.path.join(git_root, "src"), name="src")
        except Exception as exc:
            print("[train] warning: comet log_code(src) failed: {}".format(exc), flush=True)

    vst_csv = config.get("raw_vst_csv") or "data/raw/neon_vst_data_2022.csv"
    data_module = data.TreeData(
        csv_file=vst_csv,
        data_dir=config["crop_dir"],
        config=config,
        client=client,
        metadata=True,
        comet_logger=comet_logger,
    )

    comet_logger.experiment.log_parameter("train_hash", hash_pandas_object(data_module.train))
    comet_logger.experiment.log_parameter("test_hash", hash_pandas_object(data_module.test))
    comet_logger.experiment.log_parameter("num_species", data_module.num_classes)
    comet_logger.experiment.log_table("train.csv", data_module.train)
    comet_logger.experiment.log_table("test.csv", data_module.test)

    if not config["use_data_commit"]:
        comet_logger.experiment.log_table("novel_species.csv", data_module.novel)

    train = data_module.train.copy()
    test = data_module.test.copy()
    crowns = data_module.crowns.copy()

    if "individual" not in train.columns and "individualID" in train.columns:
        train["individual"] = train["individualID"]
    if "individual" not in test.columns and "individualID" in test.columns:
        test["individual"] = test["individualID"]

    train = train[~train.individual.str.contains("graves")].reset_index(drop=True)
    test = test[~test.individual.str.contains("graves")].reset_index(drop=True)

    print(
        "[train] Building MultiStage (5 levels). If preload_images is True, this loads "
        "every crop into RAM for each level and can take many minutes with no GPU use yet.",
        flush=True,
    )
    m = multi_stage.MultiStage(train, test, config=data_module.config, crowns=crowns)
    print("[train] MultiStage ready; starting Trainer.fit …", flush=True)

    for index, train_df in enumerate(
        [
            m.level_0_train,
            m.level_1_train,
            m.level_2_train,
            m.level_3_train,
            m.level_4_train,
        ]
    ):
        comet_logger.experiment.log_table("train_level_{}.csv".format(index), train_df)

    for index, test_df in enumerate(
        [
            m.level_0_test,
            m.level_1_test,
            m.level_2_test,
            m.level_3_test,
            m.level_4_test,
        ]
    ):
        comet_logger.experiment.log_table("test_level_{}.csv".format(index), test_df)

    acc, dev = utils.trainer_accelerator_devices(data_module.config)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer_kwargs: dict = {
        "accelerator": acc,
        "devices": dev,
        "fast_dev_run": data_module.config["fast_dev_run"],
        "max_epochs": data_module.config["epochs"],
        "num_sanity_val_steps": 0,
        "enable_checkpointing": False,
        "callbacks": [lr_monitor],
        "logger": comet_logger,
        "profiler": "simple",
    }
    lt = data_module.config.get("limit_train_batches")
    lv = data_module.config.get("limit_val_batches")
    lp = data_module.config.get("limit_predict_batches")
    if lt is not None:
        trainer_kwargs["limit_train_batches"] = lt
    if lv is not None:
        trainer_kwargs["limit_val_batches"] = lv
    if lp is not None:
        trainer_kwargs["limit_predict_batches"] = lp

    trainer = Trainer(**trainer_kwargs)

    trainer.fit(m)

    ckpt_dir = data_module.config.get("checkpoint_dir", "results/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    exp = comet_logger.experiment
    exp_id = getattr(exp, "id", None) or exp.get_key()
    ckpt_path = os.path.join(ckpt_dir, "{}.pt".format(exp_id))
    trainer.save_checkpoint(ckpt_path)
    print("[train] wrote checkpoint:", ckpt_path)

    print("Before prediction, the taxonID value counts")
    print(test.taxonID.value_counts())

    ds = data.TreeDataset(df=test, train=False, config=config)
    predictions = trainer.predict(m, dataloaders=m.predict_dataloader(ds))
    results = m.gather_predictions(predictions)
    results["individual"] = results["individual"]
    results_with_data = results.merge(crowns, on="individual")
    comet_logger.experiment.log_table("nested_predictions.csv", results_with_data)

    ensemble_df = m.ensemble(results)
    truth_cols = test.drop_duplicates(subset=["individual"])[["individual", "label", "siteID"]]
    ensemble_df = ensemble_df.merge(truth_cols, on="individual", how="inner")
    if ensemble_df.empty:
        print(
            "[train] warning: no individuals overlap between predictions and test labels "
            "(e.g. limit_predict_batches too small); skipping evaluation_scores."
        )
    else:
        exp_for_eval = comet_logger.experiment if isinstance(comet_logger, CometLogger) else None
        ensemble_df = m.evaluation_scores(ensemble_df, experiment=exp_for_eval)

    comet_logger.experiment.log_table("ensemble_df.csv", ensemble_df)


if __name__ == "__main__":
    main_train()
