import json
from pathlib import Path

import numpy as np
import pytest

from scripts.gap_benchmarks.diagnose_dna_anchor_fit_modes import rigid_fit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "artifacts/gap_benchmarks/dna_anchor_fit_comparison.json"


def test_rigid_fit_recovers_a_proper_row_vector_transform():
    mobile = np.array([
        [0.0, 0.0, 0.0],
        [1.2, 0.1, 0.0],
        [0.2, 1.1, 0.1],
        [0.1, 0.3, 1.3],
        [-0.3, 0.4, 0.7],
    ])
    angle = np.deg2rad(37.0)
    rotation_expected = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    translation_expected = np.array([4.2, -1.3, 8.0])
    target = mobile @ rotation_expected + translation_expected

    rotation, translation = rigid_fit(mobile, target)

    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert rotation == pytest.approx(rotation_expected, abs=1e-10)
    assert translation == pytest.approx(translation_expected, abs=1e-10)
    assert mobile @ rotation + translation == pytest.approx(target, abs=1e-10)


def test_full_anchor_fit_diagnostic_keeps_formal_limits_and_exposes_no_survivors():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["protocol"] == "masked-dna-gap-baseframe-vs-full-anchor-kabsch-diagnostic-v1"
    assert report["template_entry_ids"] == ["119D", "1BNA", "1WQY", "3SV3"]
    assert report["template_suite_count"] == 67
    assert report["target_entry"] == "1KX5"
    assert report["window_count"] == 9
    assert report["masked_middle_coordinates_used"] is False
    assert report["formal_thresholds_changed"] is False
    assert report["environment_screening"] == "not_evaluated"
    assert report["scientific_acceptance"] is False
    assert sum(row["pair_count"] for row in report["runs"]) == 164
    assert all(row["full_anchor_rigid_fit_survivor_count"] == 0 for row in report["runs"])
    assert sum(sum(row["first_failure_counts"].values()) for row in report["runs"]) == 164
    assert all(
        row["max_anchor_displacement_A"]["full_anchor_rigid_fit_residual"]["median"]
        <= row["max_anchor_displacement_A"]["baseframe_placement"]["median"]
        for row in report["runs"]
    )
    assert all(
        pair["full_anchor_rigid_fit"]["passes_existing_geometry_limits"] is False
        for row in report["runs"] for pair in row["best_full_anchor_rigid_fit_pairs"]
    )
