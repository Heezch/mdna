"""Evaluate 1KX5-derived suite closure on masked, entry-held-out 1BNA windows."""

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
from mdna.chemistry import get_nucleotide_component
from scripts.gap_benchmarks.audit_1bna_dna_structure import (
    DEFAULT_INPUT as DEFAULT_TARGET,
    DEFAULT_LIBRARY as DEFAULT_TEMPLATE_LIBRARY,
    DEFAULT_OUTPUT as DEFAULT_AUDIT,
    EXPECTED_HBONDS,
    _atom_coordinates,
    audit,
)
from scripts.gap_benchmarks.extract_1kx5_dna_suites import (
    DEFAULT_INPUT as DEFAULT_TEMPLATE_SOURCE,
    RESIDUE_TO_BASE,
    make_base_frame,
)

DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/gap_benchmarks/1bna_dna_gap_loo.json"


def _frame_for_residue(residue: Any, base: str) -> np.ndarray:
    coordinates = _atom_coordinates(residue)
    origin, column_axes = make_base_frame(base, coordinates)
    return np.vstack((origin, column_axes.T))


def _anchor_atoms(residue: Any) -> dict[str, list[float]]:
    coordinates = _atom_coordinates(residue)
    registered_heavy_atoms = {
        atom.name for atom in get_nucleotide_component(RESIDUE_TO_BASE[residue.name]).atoms
        if atom.element.upper() != "H"
    }
    missing = registered_heavy_atoms - coordinates.keys()
    if missing:
        raise ValueError(
            f"{residue.name}:{residue.seqid} anchor is missing registered heavy atoms: "
            f"{', '.join(sorted(missing))}"
        )
    return {name: coordinates[name].tolist() for name in sorted(registered_heavy_atoms)}


def _candidate_atoms(candidate: Any) -> dict[str, np.ndarray]:
    return {atom["atom_name"]: np.asarray(atom["xyz_A"], dtype=float)
            for atom in candidate.assembly.atoms if atom["residue_index"] == 1}


def _paired_hbond_distances(atoms_a: dict[str, np.ndarray], atoms_b: dict[str, np.ndarray],
                            base_pair: str) -> list[float]:
    return [float(np.linalg.norm(atoms_a[first] - atoms_b[second]))
            for first, second in EXPECTED_HBONDS[base_pair]]


def _mean_coordinate_rmsd(predicted: dict[str, np.ndarray],
                          observed: dict[str, np.ndarray]) -> float:
    common = sorted(predicted.keys() & observed.keys())
    if not common:
        raise ValueError("Gap prediction and held-out truth have no common heavy atoms")
    delta = np.stack([predicted[name] - observed[name] for name in common])
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def evaluate(
    target_path: Path = DEFAULT_TARGET,
    template_library_path: Path = DEFAULT_TEMPLATE_LIBRARY,
    audit_path: Path = DEFAULT_AUDIT,
    *,
    limits: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    import gemmi

    source_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_source_hash = hashlib.sha256(target_path.read_bytes()).hexdigest()
    if source_audit["source_sha256"] != expected_source_hash:
        raise ValueError("1BNA audit source hash does not match the target coordinate file")
    if any(source["file"].startswith("mdna/atomic/ddd/1bna")
           for source in json.loads(template_library_path.read_text())["source_files"]):
        raise ValueError("Held-out target entry 1BNA is present in the template library")

    default_limits: dict[str, float | int] = {
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
    search_limits = {**default_limits, **(limits or {})}
    library = DNAFragmentLibrary.from_json(template_library_path)
    structure = gemmi.read_structure(str(target_path))
    chains = {chain.name: [(index, residue) for index, residue in enumerate(chain)
                           if residue.name in RESIDUE_TO_BASE]
              for chain in structure[0]}
    windows = []
    for window in source_audit["candidate_paired_gap_windows"]:
        strand_results = {}
        for chain_id, missing_key, triplet_key in (
            ("A", "chain_A_missing_index_0based", "chain_A_triplet"),
            ("B", "chain_B_missing_index_0based", "chain_B_triplet"),
        ):
            missing_index = window[missing_key]
            triplet = window[triplet_key]
            residues = [residue for _, residue in chains[chain_id]]
            upstream, downstream = residues[missing_index - 1], residues[missing_index + 1]
            start_frame = _frame_for_residue(upstream, triplet[0])
            end_frame = _frame_for_residue(downstream, triplet[2])
            result = close_dna_gap_from_anchors(
                library,
                triplet,
                start_frame,
                end_frame,
                _anchor_atoms(upstream),
                _anchor_atoms(downstream),
                coordinate_unit="angstrom",
                **search_limits,
            )
            strand_results[chain_id] = result

        paired_candidates = []
        for candidate_a in strand_results["A"].candidates:
            for candidate_b in strand_results["B"].candidates:
                missing_a = _candidate_atoms(candidate_a)
                missing_b = _candidate_atoms(candidate_b)
                hbond_distances = _paired_hbond_distances(
                    missing_a, missing_b, window["missing_base_pair"]
                )
                if max(hbond_distances) > 3.5:
                    continue

                # Held-out residue coordinates are read only after both strands
                # have produced candidates and the pair has passed model-only
                # Watson-Crick contact screening.
                residues_a = [residue for _, residue in chains["A"]]
                residues_b = [residue for _, residue in chains["B"]]
                truth_a = _atom_coordinates(residues_a[window["chain_A_missing_index_0based"]])
                truth_b = _atom_coordinates(residues_b[window["chain_B_missing_index_0based"]])
                rmsd_a = _mean_coordinate_rmsd(missing_a, truth_a)
                rmsd_b = _mean_coordinate_rmsd(missing_b, truth_b)
                paired_candidates.append({
                    "suite_ids_A": list(candidate_a.suite_ids),
                    "suite_ids_B": list(candidate_b.suite_ids),
                    "combined_score": candidate_a.normalized_score + candidate_b.normalized_score,
                    "missing_base_pair_hbond_distances_A": hbond_distances,
                    "max_missing_base_pair_hbond_distance_A": max(hbond_distances),
                    "chain_A_missing_residue_heavy_atom_rmsd_A": rmsd_a,
                    "chain_B_missing_residue_heavy_atom_rmsd_A": rmsd_b,
                    "paired_missing_residue_heavy_atom_rmsd_A": float(np.sqrt(
                        (rmsd_a**2 + rmsd_b**2) / 2.0
                    )),
                })
        paired_candidates.sort(key=lambda item: (item["combined_score"],
                                                  item["suite_ids_A"], item["suite_ids_B"]))
        windows.append({
            "chain_A_missing_index_0based": window["chain_A_missing_index_0based"],
            "chain_B_missing_index_0based": window["chain_B_missing_index_0based"],
            "chain_A_triplet": window["chain_A_triplet"],
            "chain_B_triplet": window["chain_B_triplet"],
            "missing_base_pair": window["missing_base_pair"],
            "training_contexts": window["training_contexts"],
            "strands": {
                chain_id: result.to_report()
                for chain_id, result in strand_results.items()
            },
            "paired_candidate_count": len(paired_candidates),
            "paired_candidates": paired_candidates,
            "missing_base_pair_hbond_distances_A": (
                paired_candidates[0]["missing_base_pair_hbond_distances_A"]
                if paired_candidates else None
            ),
            "best_paired_candidate_rmsd_A": (
                paired_candidates[0]["paired_missing_residue_heavy_atom_rmsd_A"]
                if paired_candidates else None
            ),
            "masked_truth_read_after_search": True,
            "environment_screening": "not_evaluated",
        })

    return {
        "schema_version": 1,
        "algorithm": "1kx5-source-two-strand-single-pair-gap-1bna-entry-holdout-v1",
        "target_entry": "1BNA",
        "target_source_file": target_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "target_source_sha256": expected_source_hash,
        "template_entries": ["1KX5"],
        "template_source_file": DEFAULT_TEMPLATE_SOURCE.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "template_source_sha256": json.loads(template_library_path.read_text())["source_files"][0]["sha256"],
        "target_entry_excluded_from_templates": True,
        "masked_truth_used_for_candidate_generation": False,
        "candidate_window_count": len(windows),
        "limits": search_limits,
        "window_selection_policy": source_audit["candidate_selection_policy"],
        "windows": windows,
        "environment_screening": "not_evaluated",
        "scientific_acceptance": False,
        "interpretation_limit": "Single held-out PDB entry; candidate windows are not independent samples. Candidate generation uses only flank frames and complete heavy-atom coordinates of the two fixed anchor residues. Missing-pair coordinates are read after candidate generation for evaluation. Environment clashes, sequence-generalization, and external structural validation are not assessed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--template-library", type=Path, default=DEFAULT_TEMPLATE_LIBRARY)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-anchor-atom-displacement-A", type=float, default=None,
                        help="override this one hard limit for a recorded sensitivity run")
    args = parser.parse_args()
    limits = (None if args.max_anchor_atom_displacement_A is None else
              {"max_anchor_atom_displacement_A": args.max_anchor_atom_displacement_A})
    report = evaluate(args.target, args.template_library, args.audit, limits=limits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "target_entry": report["target_entry"],
        "candidate_window_count": report["candidate_window_count"],
        "paired_candidate_counts": [window["paired_candidate_count"] for window in report["windows"]],
        "paired_best_rmsd_A": [window["best_paired_candidate_rmsd_A"] for window in report["windows"]],
        "strand_rejections": [
            {chain_id: {
                "status": window["strands"][chain_id]["status"],
                "closures": window["strands"][chain_id]["total_compatible_closures"],
                "rejections": window["strands"][chain_id]["rejections"],
            } for chain_id in ("A", "B")}
            for window in report["windows"]
        ],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
