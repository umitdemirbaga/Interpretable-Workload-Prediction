#!/usr/bin/env python3
"""Check intended public repository files for restricted-data candidates.

If run inside Git, inspect *tracked* files (those that can be pushed).
For a release directory without Git metadata, inspect the directory contents.
This is a conservative file/path audit, not a privacy or licensing certification.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_DATA_FILES = {
    "data/news/README.md",
    "data/metadata/README.md",
    "data/processed/README.md",
    "data/processed/SHA256SUMS",
    "data/processed/All_with_group.csv",  # Published processed infrastructure dataset
}
PROTECTED_FORMATS = {".csv", ".tsv", ".parquet", ".xlsx", ".xls", ".feather", ".arrow", ".npz", ".npy", ".sqlite", ".db"}
PROTECTED_PATH_PARTS = {"private", "raw", "workloads"}


def listed_release_files(root: Path = ROOT) -> list[Path]:
    if (root / ".git").exists():
        proc = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True)
        return [Path(x.decode("utf-8")) for x in proc.stdout.split(b"\0") if x]
    return [p.relative_to(root) for p in root.rglob("*") if p.is_file() and ".git" not in p.relative_to(root).parts]


def restricted_paths(paths: list[Path]) -> list[str]:
    issues = []
    for rel in paths:
        key = rel.as_posix()
        parts = set(rel.parts)
        if key in ALLOWED_DATA_FILES:
            continue
        if "data" in parts:
            issues.append(f"Unapproved data file: {key}")
        elif key.startswith("results/"):
            issues.append(f"Unreviewed generated result: {key}")
        elif key.startswith("figures/"):
            issues.append(f"Unreviewed generated figure: {key}")
        elif rel.suffix.lower() in PROTECTED_FORMATS:
            issues.append(f"Non-whitelisted potential dataset: {key}")
        elif any(part.lower() in PROTECTED_PATH_PARTS for part in rel.parts):
            issues.append(f"Potential private-data folder: {key}")
    return issues


def main() -> int:
    paths = listed_release_files()
    issues = restricted_paths(paths)
    if issues:
        print("FAIL: Restricted or unreviewed files may be published:")
        for msg in issues:
            print(f"  - {msg}")
        return 1
    print(f"PASS: {len(paths)} release files audited; no excluded healthcare datasets present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
