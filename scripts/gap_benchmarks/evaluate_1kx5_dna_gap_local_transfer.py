"""Run a same-entry, two-suite-excluded DNA gap transfer diagnostic."""

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
from scripts.gap_benchmarks.extract_1kx5_dna_suites import RESIDUE_TO_BASE


DEFAULT_LIBRARY = PROJECT_ROOT / "artifacts/gap_benchmarks/1kx5_dna_suites.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/gap_benchmarks/1kx5_dna_gap_local_transfer.json"
TARGET_SUITE_IDS = ("1KX5:I:-72--71", "1KX5:I:-71--70")
DEFAULT_LIMITS: dict[str, float | int] = {
    "max_translation_residual_A": 0.75,
    "max_rotation_residual_degrees": 12.0,
    "max_shared_residue_rmsd_A": 0.8,
    "max_shared_atom_displacement_A": 0.8,
    "max_bond_length_deviation_A": 0.25,
    "max_anchor_atom_displacement_A": 0.8,
    "max_anchor_connection_length_deviation_A": 0.25,
    "max_search_states": 10000,
    "max_returned_candidates": 30,
}
SUGAR_PHOSPHATE_ATOMS = {
    "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "C1'",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_atoms(candidate: Any) -> dict[str, np.ndarray]:
    return {
        atom["atom_name"]: np.asarray(atom["xyz_A"], dtype=float)
        for atom in candidate.assembly.atoms
        if atom["residue_index"] == 1
    }


def _posthoc_truth_metrics(
    predicted: dict[str, np.ndarray], observed: dict[str, np.ndarray]
) -> dict[str, Any]:
    names = sorted(predicted.keys() & observed.keys())
    if not names:
        raise ValueError("Candidate and masked truth share no heavy atoms")
    delta = np.stack([predicted[name] - observed[name] for name in names])
    squared = np.sum(delta * delta, axis=1)
    groups = {
        "base": [i for i, name in enumerate(names) if name not in SUGAR_PHOSPHATE_ATOMS],
        "sugar_phosphate": [i for i, name in enumerate(names) if name in SUGAR_PHOSPHATE_ATOMS],
    }
    metrics: dict[str, Any] = {
        "heavy_atom_count": len(names),
        "heavy_atom_rmsd_A": float(np.sqrt(np.mean(squared))),
        "heavy_atom_max_displacement_A": float(np.sqrt(np.max(squared))),
        "groups": {},
        "use": "post-search analysis only; not used for candidate generation or ranking",
    }
    for group, indices in groups.items():
        if indices:
            group_squared = squared[indices]
            metrics["groups"][group] = {
                "atom_count": len(indices),
                "rmsd_A": float(np.sqrt(np.mean(group_squared))),
                "max_displacement_A": float(np.sqrt(np.max(group_squared))),
            }
    return metrics


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


def evaluate(
    library_path: Path = DEFAULT_LIBRARY,
    *,
    target_suite_ids: tuple[str, str] = TARGET_SUITE_IDS,
    additional_excluded_suite_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    import gemmi

    library_data = json.loads(library_path.read_text(encoding="utf-8"))
    original_library = DNAFragmentLibrary.from_json(library_path)
    suites_by_id = {suite["suite_id"]: suite for suite in original_library._suites}
    try:
        first, second = (suites_by_id[suite_id] for suite_id in target_suite_ids)
    except KeyError as exc:
        raise ValueError(f"Target suite is absent from the library: {exc.args[0]}") from exc
    if (first["chain_id"] != second["chain_id"]
            or first["source_residues"][1] != second["source_residues"][0]
            or first["sequence"][1] != second["sequence"][0]):
        raise ValueError("Target suites must be contiguous and share the masked nucleotide.")
    if first["source_coordinate_sha256"] != second["source_coordinate_sha256"]:
        raise ValueError("Target suites must come from the same coordinate file.")

    source_record = library_data["source_files"][0]
    target_path = PROJECT_ROOT / source_record["file"]
    source_sha256 = _sha256(target_path)
    if source_sha256 != source_record["sha256"]:
        raise ValueError("Target structure SHA-256 does not match the suite library.")
    structure = gemmi.read_structure(str(target_path))
    chain = next((candidate for candidate in structure[0]
                  if candidate.name == first["chain_id"]), None)
    if chain is None:
        raise ValueError(f"Source coordinate file lacks chain {first['chain_id']!r}.")
    residues_by_number = {
        str(residue.seqid.num): residue
        for residue in chain if residue.name in RESIDUE_TO_BASE
    }
    upstream_number = first["source_residues"][0]
    missing_number = first["source_residues"][1]
    downstream_number = second["source_residues"][1]
    try:
        upstream = residues_by_number[upstream_number]
        missing = residues_by_number[missing_number]
        downstream = residues_by_number[downstream_number]
    except KeyError as exc:
        raise ValueError(f"Source coordinate file lacks target residue {exc.args[0]!r}.") from exc

    sequence = first["sequence"] + second["sequence"][1]
    start_frame = _frame_for_residue(upstream, sequence[0])
    end_frame = _frame_for_residue(downstream, sequence[2])
    start_atoms = _anchor_atoms(upstream)
    end_atoms = _anchor_atoms(downstream)

    excluded_ids = set(target_suite_ids) | set(additional_excluded_suite_ids)
    missing_exclusions = excluded_ids - set(suites_by_id)
    if missing_exclusions:
        raise ValueError(
            f"Excluded suite IDs are absent from the library: {sorted(missing_exclusions)}"
        )
    candidate_suites = [
        suite for suite in original_library._suites
        if suite["suite_id"] not in excluded_ids
    ]
    candidate_library = DNAFragmentLibrary(
        candidate_suites,
        source_files=tuple(original_library.source_files),
    )
    candidate_suite_ids = [suite["suite_id"] for suite in candidate_library._suites]
    if not excluded_ids.isdisjoint(candidate_suite_ids):
        raise AssertionError("Masked target suites leaked into the candidate library.")

    strict_limits = dict(DEFAULT_LIMITS)
    relaxed_limits = {
        **DEFAULT_LIMITS,
        "max_shared_atom_displacement_A": 2.0,
        "max_bond_length_deviation_A": 0.30,
    }
    strict_result = _run_search(
        candidate_library, sequence, start_frame, end_frame,
        start_atoms, end_atoms, strict_limits,
    )
    relaxed_result = _run_search(
        candidate_library, sequence, start_frame, end_frame,
        start_atoms, end_atoms, relaxed_limits,
    )
    strict = strict_result.to_report()
    relaxed = relaxed_result.to_report()

    # The masked nucleotide coordinates are first read after both searches finish.
    masked_truth = _atom_coordinates(missing)
    def score_after_search(candidates: Any) -> list[dict[str, Any]]:
        metrics = []
        for rank, candidate in enumerate(candidates, start=1):
            truth_metrics = _posthoc_truth_metrics(_candidate_atoms(candidate), masked_truth)
            metrics.append({
                "solver_rank": rank,
                "suite_ids": list(candidate.suite_ids),
                "normalized_score": candidate.normalized_score,
                "missing_residue_posthoc_truth_metrics": truth_metrics,
                "missing_residue_heavy_atom_rmsd_to_masked_truth_A": truth_metrics[
                    "heavy_atom_rmsd_A"
                ],
            })
        return metrics

    strict_truth_metrics = score_after_search(strict_result.candidates)
    relaxed_truth_metrics = score_after_search(relaxed_result.candidates)
    strict["candidate_truth_metrics"] = strict_truth_metrics
    strict["best_candidate_rmsd_to_masked_truth_A"] = (
        min(item["missing_residue_heavy_atom_rmsd_to_masked_truth_A"]
            for item in strict_truth_metrics)
        if strict_truth_metrics else None
    )
    strict["truth_metric_use"] = (
        "computed after strict and relaxed candidate generation; not used for candidate selection"
    )

    return {
        "schema_version": 1,
        "protocol": "same-entry-two-suite-excluded-dna-gap-local-transfer-diagnostic-v1",
        "target_entry": first["source_entry"],
        "target_source_file": source_record["file"],
        "target_source_sha256": source_sha256,
        "suite_library_file": library_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "suite_library_sha256": _sha256(library_path),
        "target_chain": first["chain_id"],
        "target_residue_numbers_5prime_to_3prime": [
            upstream_number, missing_number, downstream_number,
        ],
        "target_sequence_including_anchors": sequence,
        "target_suite_ids": list(target_suite_ids),
        "additional_excluded_suite_ids": list(additional_excluded_suite_ids),
        "excluded_suite_ids": sorted(excluded_ids),
        "candidate_library_suite_count": len(candidate_suite_ids),
        "candidate_library_suite_ids": candidate_suite_ids,
        "exact_target_suites_excluded_from_template_library": True,
        "masked_truth_used_for_candidate_generation": False,
        "masked_truth_read_after_search": True,
        "runs": {
            "strict_defaults": strict,
            "interface_relaxed_diagnostic": {
                **relaxed,
                "candidate_truth_metrics": relaxed_truth_metrics,
                "best_candidate_rmsd_to_masked_truth_A": (
                    min(item["missing_residue_heavy_atom_rmsd_to_masked_truth_A"]
                        for item in relaxed_truth_metrics)
                    if relaxed_truth_metrics else None
                ),
                "selection_policy": (
                    "Candidates remain ordered by the solver score. Masked-truth RMSD is "
                    "computed only after candidate generation and does not select a candidate."
                ),
            },
        },
        "scientific_acceptance": False,
        "interpretation_limit": (
            "same-entry same-structure-family transfer diagnostic; the two exact target "
            "suites are excluded but neighboring suites from the same 1KX5 entry remain. "
            "The relaxed run is exploratory and exceeds the strict shared-atom and bond "
            "length limits. Neither run establishes independent-family generalization, "
            "environment compatibility, or structural acceptance."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = evaluate(args.library)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "target_entry": report["target_entry"],
        "target_suite_ids": report["target_suite_ids"],
        "strict_status": report["runs"]["strict_defaults"]["status"],
        "strict_closures": report["runs"]["strict_defaults"]["total_compatible_closures"],
        "relaxed_status": report["runs"]["interface_relaxed_diagnostic"]["status"],
        "relaxed_closures": report["runs"]["interface_relaxed_diagnostic"]["total_compatible_closures"],
        "diagnostic_rmsd_A": report["runs"]["interface_relaxed_diagnostic"]["best_candidate_rmsd_to_masked_truth_A"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
