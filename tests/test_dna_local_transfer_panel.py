import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "artifacts/gap_benchmarks/1kx5_dna_gap_local_transfer_panel.json"


def test_spaced_local_transfer_panel_preserves_holdout_and_acceptance_boundaries():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["target_entry"] == "1KX5"
    assert report["panel_window_count"] == 9
    assert report["each_window_excludes_its_two_exact_target_suites"] is True
    assert report["paired_control_also_excludes_same_index_suite_pair_from_other_chain"] is True
    assert report["masked_truth_used_for_candidate_generation"] is False
    assert report["scientific_acceptance"] is False
    assert len({tuple(row["target_suite_ids"]) for row in report["runs"]}) == 9
    strict_positive_rows = [
        row for row in report["runs"]
        if row["strict_defaults"]["total_compatible_closures"] > 0
    ]
    assert len(strict_positive_rows) == 1
    strict_positive = strict_positive_rows[0]["strict_defaults"]
    assert strict_positive["best_candidate_rmsd_to_masked_truth_A"] < 0.5
    assert strict_positive["truth_metric_use"].startswith("computed after")
    assert strict_positive_rows[0]["same_entry_same_residue_index_template_matches"] == [
        "1KX5:J:8-9", "1KX5:J:9-10"
    ]
    assert strict_positive_rows[0]["same_index_copy_excluded"]["strict_defaults"][
        "total_compatible_closures"
    ] == 0

    for row in report["runs"]:
        assert row["exact_target_suites_excluded"] is True
        assert row["masked_truth_used_for_candidate_generation"] is False
        assert row["scientific_acceptance"] is False
        assert row["candidate_library_suite_count"] == 286
        assert row["same_index_copy_excluded"]["candidate_library_suite_count"] == 284
        assert row["same_index_copy_excluded"]["additional_excluded_suite_ids"] == [
            suite_id.replace(":I:", ":J:") for suite_id in row["target_suite_ids"]
        ]
        assert row["strict_defaults"]["limits"]["max_shared_atom_displacement_A"] == 0.8
        assert row["strict_defaults"]["limits"]["max_bond_length_deviation_A"] == 0.25
        assert row["interface_relaxed_diagnostic"]["limits"][
            "max_shared_atom_displacement_A"
        ] == 2.0
        assert row["interface_relaxed_diagnostic"]["limits"][
            "max_bond_length_deviation_A"
        ] == 0.30
        for candidate_metric in row["strict_defaults"]["candidate_truth_metrics"]:
            assert candidate_metric["missing_residue_posthoc_truth_metrics"]["use"].startswith(
                "post-search analysis only"
            )
        for candidate_metric in row["interface_relaxed_diagnostic"]["candidate_truth_metrics"]:
            assert candidate_metric["missing_residue_posthoc_truth_metrics"]["use"].startswith(
                "post-search analysis only"
            )
