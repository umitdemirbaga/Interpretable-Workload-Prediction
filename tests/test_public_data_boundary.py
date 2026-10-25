from pathlib import Path
from scripts.audit_public_files import listed_release_files, restricted_paths


def test_no_restricted_healthcare_inputs_in_release():
    assert restricted_paths(listed_release_files()) == []


def test_private_data_are_rejected():
    candidates = [Path("data/news/NEWS_250K.csv"), Path("data/raw/1M.csv"),
                  Path("results/full/patient_level.csv")]
    assert len(restricted_paths(candidates)) == len(candidates)
