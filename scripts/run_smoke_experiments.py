#!/usr/bin/env python3
"""Small-scale execution checks for data preparation and model selection."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    env = os.environ.copy()
    env['EXPERIMENT_MODE'] = 'smoke'
    for prefix in ('00_', '01_', '02_', '03_'):
        path = next((ROOT / 'experiments').glob(f'{prefix}*.py'))
        subprocess.run([sys.executable, str(path)], cwd=ROOT, env=env,
                       check=True, timeout=180)
        print(f'PASS {path.name}', flush=True)


if __name__ == '__main__':
    main()
