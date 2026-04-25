"""CLI: update NEON woody vegetation structure table and compare with baseline."""

from __future__ import annotations

import argparse
import json
import os

from src import utils
from src.vst_update import update_and_compare_vst


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download NEON VST updates and compare with baseline individualIDs."
    )
    parser.add_argument("--config", default="config.yml", help="Config yaml")
    parser.add_argument(
        "--baseline-csv",
        default="data/raw/neon_vst_data_2022.csv",
        help="Baseline VST CSV used for comparison.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/raw/neon_vst_data_latest.csv",
        help="Path to write assembled latest VST rows.",
    )
    parser.add_argument(
        "--since-csv",
        default="data/raw/neon_vst_data_since_2022.csv",
        help="Path to write rows downloaded since startdate.",
    )
    parser.add_argument(
        "--summary-csv",
        default="results/vst_additions_by_site_since_2022.csv",
        help="Path to write per-site summary.",
    )
    parser.add_argument(
        "--new-ids-csv",
        default="results/vst_new_individual_ids_not_in_2022.csv",
        help="Path to write new individualIDs.",
    )
    parser.add_argument(
        "--startdate",
        default="2022-01",
        help="Earliest YYYY-MM date to request from NEON.",
    )
    parser.add_argument(
        "--sites",
        nargs="*",
        default=None,
        help="Optional site list; defaults to sites present in baseline CSV.",
    )
    parser.add_argument(
        "--token-env",
        default="NEON_API_TOKEN",
        help="Environment variable name containing NEON API token.",
    )
    parser.add_argument(
        "--exclude-provisional",
        action="store_true",
        help="Exclude provisional releases (defaults to include provisional).",
    )
    args = parser.parse_args(argv)

    config = utils.read_config(args.config)
    token = os.environ.get(args.token_env)

    result = update_and_compare_vst(
        baseline_csv=args.baseline_csv,
        startdate=args.startdate,
        sites=args.sites,
        token=token,
        include_provisional=(not args.exclude_provisional),
    )

    # Write canonical "latest" as baseline union updates.
    import pandas as pd

    old = pd.read_csv(args.baseline_csv, low_memory=False)
    latest = pd.concat([old, result.full_df], ignore_index=True, sort=False)
    if "individualID" in latest.columns and "siteID" in latest.columns and "eventID" in latest.columns:
        latest = latest.drop_duplicates(subset=["individualID", "siteID", "eventID"], keep="last")
    elif "individualID" in latest.columns and "siteID" in latest.columns:
        latest = latest.drop_duplicates(subset=["individualID", "siteID"], keep="last")

    for path in [args.output_csv, args.since_csv, args.summary_csv, args.new_ids_csv]:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    latest.to_csv(args.output_csv, index=False)
    result.full_df.to_csv(args.since_csv, index=False)
    result.summary_by_site.to_csv(args.summary_csv)
    result.new_ids_df.sort_values(["siteID", "individualID"]).to_csv(args.new_ids_csv, index=False)

    totals = {
        "baseline_rows_2022_file": int(len(old)),
        "baseline_unique_individualIDs": int(old["individualID"].astype(str).nunique()),
        "new_rows_since_startdate": int(len(result.full_df)),
        "new_unique_individualIDs_since_startdate": int(result.full_df["individualID"].astype(str).nunique()),
        "new_unique_individualIDs_not_in_2022": int(result.new_ids_df["individualID"].astype(str).nunique()),
        "failed_sites": result.failed_sites,
        "output_csv": args.output_csv,
        "since_csv": args.since_csv,
        "summary_csv": args.summary_csv,
        "new_ids_csv": args.new_ids_csv,
    }
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
