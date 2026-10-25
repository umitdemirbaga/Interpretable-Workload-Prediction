#!/usr/bin/env python3
"""Check experiment scripts and processed-dataset consistency."""
from __future__ import annotations
import hashlib
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "All_with_group.csv"
FEATURES = ["dataSplit", "map", "shuffle", "reduce", "dataCombine",
            "networkTraffic", "CPUusage", "Musage"]

def main():
    scripts = sorted((ROOT / "experiments").glob("*.py"))
    assert len(scripts) == 13, f"Expected 13 experiments; found {len(scripts)}"
    for path in scripts:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    assert not list(ROOT.rglob("*.ipynb")), "Notebook files remain in the repository."
    df = pd.read_csv(DATA)
    required = FEATURES + ['group', 'makespan', 'normalized_makespan', 'performance']
    assert set(required).issubset(df.columns)
    assert len(df) == 24000 and df.group.nunique() == 800
    assert df.groupby('group', sort=False).size().eq(30).all()
    assert df.group.eq(np.arange(len(df)) // 30 + 1).all()
    assert df.performance.value_counts().to_dict() == {1: 19030, 0: 4970}
    med = df.groupby('group').normalized_makespan.transform('median')
    labels = (df.normalized_makespan < 1.5 * med).astype(int)
    assert labels.equals(df.performance), 'Archived reference labels are inconsistent'
    digest = hashlib.sha256(DATA.read_bytes()).hexdigest()
    manifest = (ROOT / 'data' / 'processed' / 'SHA256SUMS').read_text().split()[0]
    assert digest == manifest, 'Dataset hash differs from the included manifest'
    print(f'PASS: {len(scripts)} Python experiment scripts')
    print('PASS: 24,000 records, 800 positional groups, reference class labels')
    print(f'PASS: SHA-256 {digest}')

if __name__ == '__main__':
    try:main()
    except Exception as ex:
        print(f'FAIL: {ex}', file=sys.stderr)
        raise
