"""Update and compare NEON woody vegetation structure (VST) tables."""

from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
from typing import Dict, Iterable, List

import pandas as pd

try:
    import neonutilities as nu
except ImportError as exc:  # pragma: no cover
    nu = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


VST_DPID = "DP1.10098.001"
STACK_DIR = "filesToStack10098"


@dataclass
class VstUpdateResult:
    full_df: pd.DataFrame
    summary_by_site: pd.DataFrame
    new_ids_df: pd.DataFrame
    failed_sites: List[str]


def _require_neonutilities() -> None:
    if nu is None:
        raise ImportError(
            "neonutilities is required. Install dependencies (e.g. uv sync) and retry."
        ) from _IMPORT_ERROR


def _normalize_site_list(baseline_df: pd.DataFrame, sites: Iterable[str] | None) -> List[str]:
    if sites:
        return sorted({str(x) for x in sites if str(x).strip()})
    return sorted({str(x) for x in baseline_df["siteID"].dropna().unique().tolist()})


def _to_dataframe(obj: object, key: str) -> pd.DataFrame:
    if isinstance(obj, dict):
        return obj.get(key, pd.DataFrame()).copy()
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    return pd.DataFrame()


def _site_latest_vst(
    *,
    site: str,
    startdate: str,
    token: str | None,
    include_provisional: bool,
) -> pd.DataFrame:
    # neonutilities accumulates temporary table files in a fixed folder name.
    # Clean it before each site to avoid cross-site contamination.
    shutil.rmtree(STACK_DIR, ignore_errors=True)

    obj = nu.load_by_product(
        dpid=VST_DPID,
        site=site,
        startdate=startdate,
        tabl="all",
        check_size=False,
        include_provisional=include_provisional,
        progress=False,
        token=token,
    )
    apparent = _to_dataframe(obj, "vst_apparentindividual")
    mapping = _to_dataframe(obj, "vst_mappingandtagging")
    perplot = _to_dataframe(obj, "vst_perplotperyear")

    if apparent.empty:
        return apparent

    if not mapping.empty:
        map_keep = [
            "individualID",
            "siteID",
            "plotID",
            "pointID",
            "stemDistance",
            "stemAzimuth",
            "recordType",
            "supportingStemIndividualID",
            "previouslyTaggedAs",
            "samplingProtocolVersion",
            "taxonID",
            "scientificName",
            "taxonRank",
            "identificationReferences",
            "morphospeciesID",
            "morphospeciesIDRemarks",
            "identificationQualifier",
        ]
        map_use = [x for x in map_keep if x in mapping.columns]
        if map_use:
            mapping = (
                mapping[map_use]
                .drop_duplicates(subset=["individualID", "siteID", "plotID"], keep="last")
            )
            apparent = apparent.merge(
                mapping,
                on=["individualID", "siteID", "plotID"],
                how="left",
            )

    if not perplot.empty:
        plot_keep = [
            "siteID",
            "plotID",
            "plotType",
            "subtype",
            "decimalLatitude",
            "decimalLongitude",
            "geodeticDatum",
            "utmZone",
            "easting",
            "northing",
            "coordinateUncertainty",
            "elevation",
            "elevationUncertainty",
            "nlcdClass",
        ]
        plot_use = [x for x in plot_keep if x in perplot.columns]
        if plot_use:
            perplot = perplot[plot_use].drop_duplicates(subset=["siteID", "plotID"], keep="last")
            apparent = apparent.merge(perplot, on=["siteID", "plotID"], how="left")

    # Keep compatibility aliases used by existing code paths.
    if "decimalLatitude" in apparent.columns and "latitude" not in apparent.columns:
        apparent["latitude"] = apparent["decimalLatitude"]
    if "decimalLongitude" in apparent.columns and "longitude" not in apparent.columns:
        apparent["longitude"] = apparent["decimalLongitude"]
    if "geodeticDatum" in apparent.columns and "datum" not in apparent.columns:
        apparent["datum"] = apparent["geodeticDatum"]
    if "coordinateUncertainty" in apparent.columns and "horzUncert" not in apparent.columns:
        apparent["horzUncert"] = apparent["coordinateUncertainty"]
    if "elevationUncertainty" in apparent.columns and "vertUncert" not in apparent.columns:
        apparent["vertUncert"] = apparent["elevationUncertainty"]
    if "easting" in apparent.columns and "itcEasting" not in apparent.columns:
        apparent["itcEasting"] = apparent["easting"]
    if "northing" in apparent.columns and "itcNorthing" not in apparent.columns:
        apparent["itcNorthing"] = apparent["northing"]

    return apparent


def update_and_compare_vst(
    *,
    baseline_csv: str,
    startdate: str = "2022-01",
    sites: Iterable[str] | None = None,
    token: str | None = None,
    include_provisional: bool = True,
) -> VstUpdateResult:
    """Download latest VST tables and compare against a baseline csv."""
    _require_neonutilities()

    baseline = pd.read_csv(baseline_csv, low_memory=False)
    if "individualID" not in baseline.columns or "siteID" not in baseline.columns:
        raise ValueError("Baseline CSV must include individualID and siteID columns.")
    baseline["individualID"] = baseline["individualID"].astype(str)
    baseline["siteID"] = baseline["siteID"].astype(str)
    baseline_unique = baseline[["siteID", "individualID"]].drop_duplicates()
    baseline_ids = set(baseline_unique["individualID"])

    run_sites = _normalize_site_list(baseline, sites)
    frames: List[pd.DataFrame] = []
    failed_sites: List[str] = []
    for site in run_sites:
        try:
            df = _site_latest_vst(
                site=site,
                startdate=startdate,
                token=token,
                include_provisional=include_provisional,
            )
            if df.empty:
                df = pd.DataFrame(columns=["siteID", "individualID"])
            if "siteID" not in df.columns:
                df["siteID"] = site
            if "individualID" not in df.columns:
                df["individualID"] = pd.Series(dtype=str)
            df["siteID"] = df["siteID"].astype(str)
            df["individualID"] = df["individualID"].astype(str)
            df = df[df["siteID"] == site]
            frames.append(df)
        except Exception:
            failed_sites.append(site)
            frames.append(pd.DataFrame(columns=["siteID", "individualID"]))

    full_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["siteID", "individualID"])
    full_df["siteID"] = full_df["siteID"].astype(str)
    full_df["individualID"] = full_df["individualID"].astype(str)
    full_unique = full_df[["siteID", "individualID"]].drop_duplicates()

    new_ids_df = full_unique[~full_unique["individualID"].isin(baseline_ids)].copy()
    summary = pd.DataFrame(
        {
            "baseline_rows_2022": baseline.groupby("siteID").size(),
            "new_rows_since_2022": full_df.groupby("siteID").size(),
            "baseline_unique_individuals_2022": baseline_unique.groupby("siteID").size(),
            "new_unique_individuals_since_2022": full_unique.groupby("siteID").size(),
            "new_unique_individualIDs_not_in_2022": new_ids_df.groupby("siteID").size(),
        }
    ).fillna(0).astype(int).sort_values("new_unique_individualIDs_not_in_2022", ascending=False)

    return VstUpdateResult(
        full_df=full_df,
        summary_by_site=summary,
        new_ids_df=new_ids_df,
        failed_sites=failed_sites,
    )
