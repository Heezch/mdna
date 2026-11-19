import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "artifacts/gap_benchmarks/1bna_to_1kx5_dna_gap_panel.json"
EXPANDED_REPORT_PATH = PROJECT_ROOT / "artifacts/gap_benchmarks/1bna_3sv3_to_1kx5_dna_gap_panel.json"
THREE_SOURCE_REPORT_PATH = PROJECT_ROOT / "artifacts/gap_benchmarks/1bna_3sv3_1wqy_to_1kx5_dna_gap_panel.json"
FOUR_SOURCE_REPORT_PATH = PROJECT_ROOT / "artifacts/gap_benchmarks/1bna_3sv3_1wqy_119d_to_1kx5_dna_gap_panel.json"


def test_cross_entry_panel_has_disjoint_sources_and_explicit_template_coverage():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["template_entry_ids"] == ["1BNA"]
    assert report["target_entry"] == "1KX5"
    assert report["window_count"] == 9
    assert report["scientific_acceptance"] is False
    assert report["masked_truth_used_for_candidate_generation"] is False
    assert report["template_source_files"][0]["sha256"] != report[
        "target_coordinate_sha256"
    ]
    assert len(report["target_windows"]) == 9
    assert len(report["runs"]) == 9
    evaluated_rows = [row for row in report["runs"] if row["status"] == "evaluated"]
    unsupported_rows = [
        row for row in report["runs"] if row["status"] == "unsupported_template_context"
    ]
    assert len(evaluated_rows) == 2
    assert len(unsupported_rows) == 7
    assert {row["target_sequence_including_anchors"] for row in evaluated_rows} == {
        "GAT", "GAA"
    }

    for row in report["runs"]:
        assert row["target_entry_absent_from_template_library"] is True
        assert row["target_suite_ids_absent_from_template_library"] is True
        assert row["masked_truth_used_for_candidate_generation"] is False
        assert row["scientific_acceptance"] is False
    for row in unsupported_rows:
        assert row["strict_defaults"] is None
        assert row["interface_relaxed_diagnostic"] is None
        assert len(row["unsupported_template_contexts"]) > 0
    for row in evaluated_rows:
        assert row["unsupported_template_contexts"] == []
        assert row["strict_defaults"]["limits"]["max_shared_atom_displacement_A"] == 0.8
        assert row["interface_relaxed_diagnostic"]["limits"][
            "max_shared_atom_displacement_A"
        ] == 2.0
        assert row["early_gate_sensitivity"]["limits"][
            "max_rotation_residual_degrees"
        ] == 20.0
        assert row["early_gate_sensitivity"]["limits"][
            "max_anchor_atom_displacement_A"
        ] == 1.25
        for run_name in (
            "strict_defaults", "interface_relaxed_diagnostic", "early_gate_sensitivity"
        ):
            for candidate in row[run_name]["candidate_truth_metrics"]:
                assert candidate["missing_residue_posthoc_truth_metrics"]["use"].startswith(
                    "post-search analysis only"
                )


def test_3sv3_source_expands_context_coverage_without_claiming_family_validation():
    report = json.loads(EXPANDED_REPORT_PATH.read_text(encoding="utf-8"))

    assert report["template_entry_ids"] == ["1BNA", "3SV3"]
    assert report["target_entry"] == "1KX5"
    assert report["template_suite_count"] == 35
    assert report["window_count"] == 9
    assert report["masked_truth_used_for_candidate_generation"] is False
    assert report["scientific_acceptance"] is False
    assert "does not establish independent-family generalization" in report[
        "interpretation_limit"
    ]
    evaluated = [row for row in report["runs"] if row["status"] == "evaluated"]
    unsupported = [
        row for row in report["runs"] if row["status"] == "unsupported_template_context"
    ]
    assert len(evaluated) == 6
    assert len(unsupported) == 3
    assert {row["target_sequence_including_anchors"] for row in unsupported} == {
        "TAT", "CAG", "CTG"
    }
    assert all(row["scientific_acceptance"] is False for row in report["runs"])


def test_1wqy_adds_template_coverage_but_does_not_yet_close_any_panel_window():
    report = json.loads(THREE_SOURCE_REPORT_PATH.read_text(encoding="utf-8"))

    assert report["template_entry_ids"] == ["1BNA", "1WQY", "3SV3"]
    assert report["template_suite_count"] == 49
    assert report["target_entry"] == "1KX5"
    assert report["window_count"] == 9
    assert report["scientific_acceptance"] is False
    assert report["masked_truth_used_for_candidate_generation"] is False
    assert "does not establish independent-family generalization" in report[
        "interpretation_limit"
    ]

    evaluated = [row for row in report["runs"] if row["status"] == "evaluated"]
    unsupported = [
        row for row in report["runs"] if row["status"] == "unsupported_template_context"
    ]
    assert len(evaluated) == 7
    assert len(unsupported) == 2
    assert {row["target_sequence_including_anchors"] for row in unsupported} == {
        "CAG", "CTG"
    }
    assert all(row["strict_defaults"]["total_compatible_closures"] == 0 for row in evaluated)
    assert all(row["interface_relaxed_diagnostic"]["total_compatible_closures"] == 0
               for row in evaluated)
    assert all(row["early_gate_sensitivity"]["total_compatible_closures"] == 0
               for row in evaluated)
    assert all(row["scientific_acceptance"] is False for row in report["runs"])


def test_119d_completes_context_coverage_but_all_masked_windows_remain_unclosed():
    report = json.loads(FOUR_SOURCE_REPORT_PATH.read_text(encoding="utf-8"))

    assert report["template_entry_ids"] == ["119D", "1BNA", "1WQY", "3SV3"]
    assert report["template_suite_count"] == 67
    assert report["target_entry"] == "1KX5"
    assert report["window_count"] == 9
    assert report["scientific_acceptance"] is False
    assert report["masked_truth_used_for_candidate_generation"] is False
    assert "does not establish independent-family generalization" in report[
        "interpretation_limit"
    ]
    assert all(row["status"] == "evaluated" for row in report["runs"])
    assert all(row["unsupported_template_contexts"] == [] for row in report["runs"])
    for row in report["runs"]:
        assert row["target_entry_absent_from_template_library"] is True
        assert row["target_suite_ids_absent_from_template_library"] is True
        assert row["scientific_acceptance"] is False
        for run_name in (
            "strict_defaults", "interface_relaxed_diagnostic", "early_gate_sensitivity"
        ):
            assert row[run_name]["total_compatible_closures"] == 0
