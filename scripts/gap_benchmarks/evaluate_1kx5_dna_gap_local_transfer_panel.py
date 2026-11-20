"""Evaluate a fixed, evenly spaced panel of same-entry DNA gap windows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.gap_benchmarks.evaluate_1kx5_dna_gap_local_transfer import (
    DEFAULT_LIBRARY,
    TARGET_SUITE_IDS,
    evaluate,
)

DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/gap_benchmarks/1kx5_dna_gap_local_transfer_panel.json"
PANEL_TARGET_SUITE_IDS = (
    TARGET_SUITE_IDS,
    ("1KX5:I:-56--55", "1KX5:I:-55--54"),
    ("1KX5:I:-40--39", "1KX5:I:-39--38"),
    ("1KX5:I:-24--23", "1KX5:I:-23--22"),
    ("1KX5:I:-8--7", "1KX5:I:-7--6"),
    ("1KX5:I:8-9", "1KX5:I:9-10"),
    ("1KX5:I:24-25", "1KX5:I:25-26"),
    ("1KX5:I:40-41", "1KX5:I:41-42"),
    ("1KX5:I:56-57", "1KX5:I:57-58"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_panel(library_path: Path = DEFAULT_LIBRARY) -> dict[str, Any]:
    rows = []
    for target_suite_ids in PANEL_TARGET_SUITE_IDS:
        report = evaluate(library_path, target_suite_ids=target_suite_ids)
        same_index_copy_suite_ids = tuple(
            suite_id.replace(":I:", ":J:") for suite_id in target_suite_ids
        )
        copy_excluded_report = evaluate(
            library_path,
            target_suite_ids=target_suite_ids,
            additional_excluded_suite_ids=same_index_copy_suite_ids,
        )
        target_spans = {suite_id.rsplit(":", 1)[1] for suite_id in target_suite_ids}
        selected_candidates = (
            report["runs"]["strict_defaults"]["candidate_truth_metrics"]
            + report["runs"]["interface_relaxed_diagnostic"]["candidate_truth_metrics"]
        )
        same_index_matches = sorted({
            suite_id
            for candidate in selected_candidates
            for suite_id in candidate["suite_ids"]
            if suite_id.rsplit(":", 1)[1] in target_spans
            and suite_id.split(":")[1] != "I"
        })
        rows.append({
            "target_suite_ids": list(target_suite_ids),
            "target_residue_numbers_5prime_to_3prime": report[
                "target_residue_numbers_5prime_to_3prime"
            ],
            "target_sequence_including_anchors": report["target_sequence_including_anchors"],
            "candidate_library_suite_count": report["candidate_library_suite_count"],
            "exact_target_suites_excluded": report[
                "exact_target_suites_excluded_from_template_library"
            ],
            "masked_truth_used_for_candidate_generation": report[
                "masked_truth_used_for_candidate_generation"
            ],
            "same_entry_same_residue_index_template_matches": same_index_matches,
            "same_index_copy_excluded": {
                "additional_excluded_suite_ids": list(same_index_copy_suite_ids),
                "candidate_library_suite_count": copy_excluded_report[
                    "candidate_library_suite_count"
                ],
                "strict_defaults": copy_excluded_report["runs"]["strict_defaults"],
                "interface_relaxed_diagnostic": copy_excluded_report["runs"][
                    "interface_relaxed_diagnostic"
                ],
            },
            "strict_defaults": report["runs"]["strict_defaults"],
            "interface_relaxed_diagnostic": report["runs"][
                "interface_relaxed_diagnostic"
            ],
            "scientific_acceptance": report["scientific_acceptance"],
        })

    return {
        "schema_version": 1,
        "protocol": "fixed-spaced-same-entry-dna-gap-transfer-panel-v1",
        "target_entry": "1KX5",
        "target_chain": "I",
        "panel_selection_policy": (
            "Nine contiguous three-nucleotide windows selected at 16-residue intervals from the "
            "1KX5 chain-I suite register, before panel evaluation; this is a within-entry "
            "diagnostic panel, not an independent-family validation set."
        ),
        "panel_window_count": len(rows),
        "suite_library_file": library_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "suite_library_sha256": _sha256(library_path),
        "each_window_excludes_its_two_exact_target_suites": True,
        "paired_control_also_excludes_same_index_suite_pair_from_other_chain": True,
        "masked_truth_used_for_candidate_generation": False,
        "runs": rows,
        "scientific_acceptance": False,
        "interpretation_limit": (
            "All windows and remaining candidate templates come from the same 1KX5 entry "
            "and nucleosome structure family. Relaxed runs exceed the strict shared-atom "
            "and bond-deviation limits. Results do not establish independent-family "
            "generalization, environment compatibility, or structural acceptance."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = evaluate_panel(args.library)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "target_entry": report["target_entry"],
        "panel_window_count": report["panel_window_count"],
        "strict_statuses": [row["strict_defaults"]["status"] for row in report["runs"]],
        "strict_closure_counts": [
            row["strict_defaults"]["total_compatible_closures"] for row in report["runs"]
        ],
        "relaxed_closure_counts": [
            row["interface_relaxed_diagnostic"]["total_compatible_closures"]
            for row in report["runs"]
        ],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
