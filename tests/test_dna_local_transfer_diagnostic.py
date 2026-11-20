import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "artifacts/gap_benchmarks/1kx5_dna_gap_local_transfer.json"


def test_local_transfer_diagnostic_excludes_target_suites_and_labels_relaxed_result():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

    assert report["target_entry"] == "1KX5"
    assert report["masked_truth_used_for_candidate_generation"] is False
    assert report["exact_target_suites_excluded_from_template_library"] is True
    assert set(report["target_suite_ids"]).isdisjoint(report["candidate_library_suite_ids"])
    assert report["scientific_acceptance"] is False

    strict = report["runs"]["strict_defaults"]
    exploratory = report["runs"]["interface_relaxed_diagnostic"]
    assert strict["status"] == "no_solution_found"
    assert strict["total_compatible_closures"] == 0
    assert exploratory["status"] == "complete"
    assert exploratory["total_compatible_closures"] > 0
    assert exploratory["limits"]["max_shared_atom_displacement_A"] > strict[
        "limits"]["max_shared_atom_displacement_A"]
    assert exploratory["limits"]["max_bond_length_deviation_A"] > strict[
        "limits"]["max_bond_length_deviation_A"]
    assert exploratory["best_candidate_rmsd_to_masked_truth_A"] < 1.0
    assert report["interpretation_limit"].startswith("same-entry")
    first_candidate = exploratory["candidate_truth_metrics"][0]
    posthoc = first_candidate["missing_residue_posthoc_truth_metrics"]
    assert posthoc["use"].startswith("post-search analysis only")
    assert posthoc["groups"]["base"]["atom_count"] > 0
    assert posthoc["groups"]["sugar_phosphate"]["atom_count"] > 0
    assert posthoc["heavy_atom_rmsd_A"] == first_candidate[
        "missing_residue_heavy_atom_rmsd_to_masked_truth_A"
    ]
