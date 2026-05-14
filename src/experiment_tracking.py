"""Git + Comet helpers for reproducible experiment tracking (no manual branch/sha CLI)."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from typing import Any


def _run_git(repo_root: str, *args: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", *args],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def git_metadata(repo_root: str | None = None) -> dict[str, Any]:
    """Return branch, SHAs, dirty flag, and optional short diff (for Comet / SLURM logs)."""
    root = repo_root or os.getcwd()
    sha = _run_git(root, "rev-parse", "HEAD")
    short = _run_git(root, "rev-parse", "--short", "HEAD")
    branch = _run_git(root, "rev-parse", "--abbrev-ref", "HEAD")
    dirty = False
    diff_head = None
    if sha:
        try:
            subprocess.check_call(
                ["git", "diff", "--quiet"],
                cwd=root,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            dirty = True
            try:
                diff_head = subprocess.check_output(
                    ["git", "diff", "HEAD"],
                    cwd=root,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                if len(diff_head) > 400_000:
                    diff_head = diff_head[:400_000] + "\n\n[truncated]\n"
            except (subprocess.CalledProcessError, OSError):
                diff_head = None
    return {
        "git_branch": branch or "unknown",
        "git_sha": sha or "unknown",
        "git_short_sha": short or "unknown",
        "git_dirty": dirty,
        "git_diff_head": diff_head,
    }


def default_experiment_name(meta: dict[str, Any]) -> str:
    """Human-readable default when user does not pass --experiment-name."""
    slurm = os.environ.get("SLURM_JOB_ID")
    if slurm:
        job = os.environ.get("SLURM_JOB_NAME", "job")
        return f"{job}-{slurm}"
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d-%H%M%S")
    short = meta.get("git_short_sha") or "unknown"
    return f"{short}-{stamp}"


def comet_display_name(
    explicit: str | None,
    meta: dict[str, Any],
) -> str:
    """Resolve Comet experiment display name (``name=`` on CometLogger)."""
    for key in ("DEEPTREE_EXPERIMENT_NAME", "COMET_EXPERIMENT_NAME"):
        v = os.environ.get(key)
        if v:
            return v
    if explicit:
        return explicit
    return default_experiment_name(meta)
