"""Test cross-entry DNA gap transfer from 1BNA suites to masked 1KX5 windows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mdna import DNAFragmentLibrary, close_dna_gap_from_anchors
from scripts.gap_benchmarks.audit_1bna_dna_structure import _atom_coordinates
from scripts.gap_benchmarks.evaluate_1bna_dna_gap_loo import _anchor_atoms, _frame_for_residue
from scripts.gap_benchmarks.evaluate_1kx5_dna_gap_local_transfer import (
    DEFAULT_LIMITS,
    _candidate_atoms,
    _posthoc_truth_metrics,
)
from scripts.gap_benchmarks.evaluate_1kx5_dna_gap_local_transfer_panel import (
    PANEL_TARGET_SUITE_IDS,
)
from scripts.gap_benchmarks.extract_1kx5_dna_suites import RESIDUE_TO_BASE

DEFAULT_TEMPLATE_LIBRARY = PROJECT_ROOT / "artifacts/gap_benchmarks/1bna_dna_suites.json"
DEFAULT_TARGET_LIBRARY = PROJECT_ROOT / "artifacts/gap_benchmarks/1kx5_dna_suites.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/gap_benchmarks/1bna_to_1kx5_dna_gap_panel.json"
TARGET_WINDOWS = PANEL_TARGET_SUITE_IDS


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_search(
    library: DNAFragmentLibrary,
    sequence: str,
    start_frame: np.ndarray,
    end_frame: np.ndarray,
    start_atoms: dict[str, list[float]],
    end_atoms: dict[str, list[float]],
    limits: dict[str, float | int],
) -> Any:
    return close_dna_gap_from_anchors(
        library,
        sequence,
        start_frame,
        end_frame,
        start_atoms,
        end_atoms,
        coordinate_unit="angstrom",
        **limits,
    )


def _score_after_search(candidates: Any, truth: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for rank, candidate in enumerate(candidates, start=1):
        metrics = _posthoc_truth_metrics(_candidate_atoms(candidate), truth)
        rows.append({
            "solver_rank": rank,
            "suite_ids": list(candidate.suite_ids),
            "normalized_score": candidate.normalized_score,
            "missing_residue_posthoc_truth_metrics": metrics,
            "missing_residue_heavy_atom_rmsd_to_masked_truth_A": metrics["heavy_atom_rmsd_A"],
        })
    return rows


def evaluate_panel(
    template_library_path: Path = DEFAULT_TEMPLATE_LIBRARY,
    target_library_path: Path = DEFAULT_TARGET_LIBRARY,
    additional_template_library_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    import gemmi

    target_data = json.loads(target_library_path.read_text(encoding="utf-8"))
    template_paths = (template_library_path, *additional_template_library_paths)
    template_libraries = [DNAFragmentLibrary.from_json(path) for path in template_paths]
    target_library = DNAFragmentLibrary.from_json(target_library_path)
    template_suites = [suite for library in template_libraries for suite in library._suites]
    template_sources_by_key = {
        (source["file"], source["sha256"]): dict(source)
        for library in template_libraries for source in library.source_files
    }
    template_sources = tuple(template_sources_by_key.values())
    template_library = DNAFragmentLibrary(template_suites, source_files=template_sources)
    template_entry_ids = sorted({suite["source_entry"] for suite in template_suites})
    target_entry_ids = sorted({suite["source_entry"] for suite in target_library._suites})
    target_suites_by_id = {suite["suite_id"]: suite for suite in target_library._suites}
    template_suites_by_id = {suite["suite_id"]: suite for suite in template_library._suites}

    if target_entry_ids != ["1KX5"]:
        raise ValueError(f"Expected only 1KX5 target suites, found {target_entry_ids}.")
    if "1KX5" in template_entry_ids:
        raise ValueError("Target entry 1KX5 must not occur in the template library.")
    template_hashes = {source["sha256"] for source in template_sources}
    target_hashes = {source["sha256"] for source in target_library.source_files}
    if template_hashes & target_hashes:
        raise ValueError("Template and target coordinate sources must be disjoint.")

    for source in template_sources:
        path = PROJECT_ROOT / source["file"]
        if _sha256(path) != source["sha256"]:
            raise ValueError(f"Template source SHA-256 mismatch: {source['file']}.")
    target_source = target_data["source_files"][0]
    target_path = PROJECT_ROOT / target_source["file"]
    target_sha256 = _sha256(target_path)
    if target_sha256 != target_source["sha256"]:
        raise ValueError("Target 1KX5 structure SHA-256 does not match its suite library.")
    structure = gemmi.read_structure(str(target_path))

    rows = []
    for target_suite_ids in TARGET_WINDOWS:
        try:
            first, second = (target_suites_by_id[suite_id] for suite_id in target_suite_ids)
        except KeyError as exc:
            raise ValueError(f"Target suite is absent from 1KX5 library: {exc.args[0]}") from exc
        if any(suite_id in template_suites_by_id for suite_id in target_suite_ids):
            raise AssertionError("Target suite IDs unexpectedly occur in the template library.")
        if (first["chain_id"] != second["chain_id"]
                or first["source_residues"][1] != second["source_residues"][0]
                or first["sequence"][1] != second["sequence"][0]):
            raise ValueError("Target suites must be contiguous and share the masked nucleotide.")

        chain_id = first["chain_id"]
        chain = next((item for item in structure[0] if item.name == chain_id), None)
        if chain is None:
            raise ValueError(f"Target structure lacks chain {chain_id!r}.")
        residues = {
            str(residue.seqid.num): residue
            for residue in chain if residue.name in RESIDUE_TO_BASE
        }
        upstream_number = first["source_residues"][0]
        missing_number = first["source_residues"][1]
        downstream_number = second["source_residues"][1]
        try:
            upstream = residues[upstream_number]
            missing = residues[missing_number]
            downstream = residues[downstream_number]
        except KeyError as exc:
            raise ValueError(f"Target residue {exc.args[0]!r} is absent from 1KX5 coordinates.") from exc

        sequence = first["sequence"] + second["sequence"][1]
        target_contexts = [sequence[:2], sequence[1:]]
        available_contexts = set(template_library.available_sequences)
        unsupported_contexts = sorted(set(target_contexts) - available_contexts)
        common_row = {
            "target_suite_ids": list(target_suite_ids),
            "target_residue_numbers_5prime_to_3prime": [
                upstream_number, missing_number, downstream_number,
            ],
            "target_sequence_including_anchors": sequence,
            "required_dinucleotide_contexts": target_contexts,
            "unsupported_template_contexts": unsupported_contexts,
            "candidate_template_suite_count": len(template_library._suites),
            "target_entry_absent_from_template_library": True,
            "target_suite_ids_absent_from_template_library": True,
            "masked_truth_used_for_candidate_generation": False,
            "scientific_acceptance": False,
        }
        if unsupported_contexts:
            rows.append({
                **common_row,
                "status": "unsupported_template_context",
                "strict_defaults": None,
                "interface_relaxed_diagnostic": None,
            })
            continue

        start_frame = _frame_for_residue(upstream, sequence[0])
        end_frame = _frame_for_residue(downstream, sequence[2])
        start_atoms = _anchor_atoms(upstream)
        end_atoms = _anchor_atoms(downstream)
        relaxed_limits = {
            **DEFAULT_LIMITS,
            "max_shared_atom_displacement_A": 2.0,
            "max_bond_length_deviation_A": 0.30,
        }
        early_gate_limits = {
            **DEFAULT_LIMITS,
            "max_rotation_residual_degrees": 20.0,
            "max_anchor_atom_displacement_A": 1.25,
        }
        strict_result = _run_search(
            template_library, sequence, start_frame, end_frame,
            start_atoms, end_atoms, dict(DEFAULT_LIMITS),
        )
        relaxed_result = _run_search(
            template_library, sequence, start_frame, end_frame,
            start_atoms, end_atoms, relaxed_limits,
        )
        early_gate_result = _run_search(
            template_library, sequence, start_frame, end_frame,
            start_atoms, end_atoms, early_gate_limits,
        )

        # Do not inspect the masked residue coordinates until all searches have completed.
        truth = _atom_coordinates(missing)
        strict_metrics = _score_after_search(strict_result.candidates, truth)
        relaxed_metrics = _score_after_search(relaxed_result.candidates, truth)
        early_gate_metrics = _score_after_search(early_gate_result.candidates, truth)
        strict_report = strict_result.to_report()
        strict_report["candidate_truth_metrics"] = strict_metrics
        strict_report["best_candidate_rmsd_to_masked_truth_A"] = min(
            (item["missing_residue_heavy_atom_rmsd_to_masked_truth_A"]
             for item in strict_metrics), default=None
        )
        relaxed_report = relaxed_result.to_report()
        relaxed_report["candidate_truth_metrics"] = relaxed_metrics
        relaxed_report["best_candidate_rmsd_to_masked_truth_A"] = min(
            (item["missing_residue_heavy_atom_rmsd_to_masked_truth_A"]
             for item in relaxed_metrics), default=None
        )
        early_gate_report = early_gate_result.to_report()
        early_gate_report["candidate_truth_metrics"] = early_gate_metrics
        early_gate_report["best_candidate_rmsd_to_masked_truth_A"] = min(
            (item["missing_residue_heavy_atom_rmsd_to_masked_truth_A"]
             for item in early_gate_metrics), default=None
        )
        early_gate_report["interpretation"] = (
            "Exploratory sensitivity only: endpoint rotation and anchor-atom limits were "
            "relaxed to expose which later gates reject candidates. Not a valid closure threshold."
        )
        rows.append({
            **common_row,
            "status": "evaluated",
            "strict_defaults": strict_report,
            "interface_relaxed_diagnostic": relaxed_report,
            "early_gate_sensitivity": early_gate_report,
        })

    return {
        "schema_version": 1,
        "protocol": "cross-entry-multi-template-to-masked-1kx5-dna-gap-panel-v1",
        "template_entry_ids": template_entry_ids,
        "target_entry": "1KX5",
        "target_chain": "I",
        "template_library_files": [
            path.resolve().relative_to(PROJECT_ROOT).as_posix() for path in template_paths
        ],
        "template_library_file_hashes": [_sha256(path) for path in template_paths],
        "template_suite_count": len(template_suites),
        "template_source_files": [dict(source) for source in template_sources],
        "target_library_file": target_library_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "target_library_sha256": _sha256(target_library_path),
        "target_coordinate_file": target_source["file"],
        "target_coordinate_sha256": target_sha256,
        "target_windows": [list(pair) for pair in TARGET_WINDOWS],
        "window_count": len(rows),
        "masked_truth_used_for_candidate_generation": False,
        "runs": rows,
        "scientific_acceptance": False,
        "interpretation_limit": (
            "This is cross-entry transfer from one 1BNA structure to two sequence-covered "
            "windows among nine fixed windows in one 1KX5 target entry. It reduces direct same-entry leakage but does "
            "not establish independent-family generalization, environment compatibility, "
            "or structural acceptance. Relaxed thresholds are diagnostic only."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-library", type=Path, default=DEFAULT_TEMPLATE_LIBRARY)
    parser.add_argument("--additional-template-library", type=Path, action="append", default=[])
    parser.add_argument("--target-library", type=Path, default=DEFAULT_TARGET_LIBRARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = evaluate_panel(
        args.template_library, args.target_library, tuple(args.additional_template_library)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "templates": report["template_entry_ids"],
        "target": report["target_entry"],
        "windows": report["window_count"],
        "statuses": [
            row["strict_defaults"]["status"] if row["status"] == "evaluated" else row["status"]
            for row in report["runs"]
        ],
        "strict_closures": [
            row["strict_defaults"]["total_compatible_closures"]
            if row["status"] == "evaluated" else 0
            for row in report["runs"]
        ],
        "relaxed_closures": [
            row["interface_relaxed_diagnostic"]["total_compatible_closures"]
            if row["status"] == "evaluated" else 0
            for row in report["runs"]
        ],
        "early_gate_sensitivity_closures": [
            row["early_gate_sensitivity"]["total_compatible_closures"]
            if row["status"] == "evaluated" else 0
            for row in report["runs"]
        ],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
