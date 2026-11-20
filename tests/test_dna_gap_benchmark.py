import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = PROJECT_ROOT / "artifacts/gap_benchmarks/1bna_dna_gap_loo.json"


def test_1bna_gap_benchmark_is_entry_held_out_and_masks_truth_during_search():
    report = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))

    assert report["target_entry"] == "1BNA"
    assert report["template_entries"] == ["1KX5"]
    assert report["target_entry_excluded_from_templates"] is True
    assert report["masked_truth_used_for_candidate_generation"] is False
    assert report["candidate_window_count"] == 4
    assert report["scientific_acceptance"] is False
    assert all(window["masked_truth_read_after_search"] for window in report["windows"])
    assert all(set(window["strands"]) == {"A", "B"} for window in report["windows"])
    assert all("missing_base_pair_hbond_distances_A" in window for window in report["windows"])
    assert all(window["paired_candidate_count"] == 0 for window in report["windows"])
    first_strand = report["windows"][0]["strands"]["A"]
    assert first_strand["rejections"]["start_anchor_displacement"] == first_strand["expanded_states"]
    assert first_strand["trial_metric_minima"]["start_anchor_max_displacement_A"] > report[
        "limits"]["max_anchor_atom_displacement_A"]
    assert "endpoint_translation_residual_A" not in first_strand["trial_metric_minima"]
