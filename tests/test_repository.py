from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_experiment_scripts_compile():
    paths = sorted((ROOT / 'experiments').glob('*.py'))
    assert len(paths) == 13
    assert not list(ROOT.rglob('*.ipynb'))
    for path in paths:
        compile(path.read_text(encoding='utf-8'), str(path), 'exec')
