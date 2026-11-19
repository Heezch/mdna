"""Compare base-frame placement with full-anchor rigid fits on masked DNA gaps.

The diagnostic reads only the two observed anchor residues from each target
window. Coordinates for the masked middle residue are never used. A surviving
row is a geometric pre-screen only; environment clashes and experimental
acceptance are not evaluated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mdna import DNAFragmentLibrary
from scripts.gap_benchmarks.evaluate_1bna_dna_gap_loo import _anchor_atoms, _frame_for_residue
from scripts.gap_benchmarks.evaluate_1kx5_dna_gap_local_transfer import DEFAULT_LIMITS
from scripts.gap_benchmarks.evaluate_1kx5_dna_gap_local_transfer_panel import PANEL_TARGET_SUITE_IDS
from scripts.gap_benchmarks.extract_1kx5_dna_suites import make_base_frame, RESIDUE_TO_BASE


TEMPLATE_LIBRARIES = (
    PROJECT_ROOT / "artifacts/gap_benchmarks/1bna_dna_suites.json",
    PROJECT_ROOT / "artifacts/gap_benchmarks/3sv3_dna_suites.json",
    PROJECT_ROOT / "artifacts/gap_benchmarks/1wqy_dna_suites.json",
    PROJECT_ROOT / "artifacts/gap_benchmarks/119d_dna_suites.json",
)
TARGET_LIBRARY = PROJECT_ROOT / "artifacts/gap_benchmarks/1kx5_dna_suites.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/gap_benchmarks/dna_anchor_fit_comparison.json"


def rigid_fit(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return proper row-vector rotation and translation mapping mobile to target."""
    mobile = np.asarray(mobile, dtype=float)
    target = np.asarray(target, dtype=float)
    if mobile.shape != target.shape or mobile.ndim != 2 or mobile.shape[1] != 3 or len(mobile) < 3:
        raise ValueError("mobile and target must be matching (n, 3) arrays with n >= 3")
    if not np.isfinite(mobile).all() or not np.isfinite(target).all():
        raise ValueError("mobile and target coordinates must be finite")
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (mobile - mobile_center).T @ (target - target_center)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_t
    translation = target_center - mobile_center @ rotation
    return rotation, translation


def _suite_atom_map(placement: Any, residue_index: int) -> dict[str, np.ndarray]:
    return {
        atom["atom_name"]: np.asarray(atom["xyz_A"], dtype=float)
        for atom in placement.atoms if atom["residue_index"] == residue_index
    }


def _transform_placement(placement: Any, rotation: np.ndarray,
                         translation: np.ndarray) -> dict[tuple[int, str], np.ndarray]:
    return {
        (atom["residue_index"], atom["atom_name"]):
            np.asarray(atom["xyz_A"], dtype=float) @ rotation + translation
        for atom in placement.atoms
    }


def _frame_from_atom_map(base: str, atoms: dict[str, np.ndarray]) -> np.ndarray:
    origin, column_axes = make_base_frame(base, atoms)
    return np.vstack((origin, column_axes.T))


def _frame_residual(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    translation = float(np.linalg.norm(first[0] - second[0]))
    relative = first[1:] @ second[1:].T
    cosine = np.clip((np.trace(relative) - 1) / 2, -1.0, 1.0)
    rotation = float(np.degrees(np.arccos(cosine)))
    return translation, rotation


def _fit_metrics(fitted: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    displacement = np.linalg.norm(fitted - target, axis=1)
    return float(np.sqrt(np.mean(displacement**2))), float(displacement.max())


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(array.max()),
    }


def _downstream_fit_frame(suite: dict[str, Any], downstream: np.ndarray) -> np.ndarray:
    relative = suite["downstream_base_frame_relative_to_upstream"]
    rotation = np.asarray(relative["rotation_matrix"], dtype=float)
    translation = np.asarray(relative["translation_A"], dtype=float)
    upstream_axes = rotation @ downstream[1:]
    upstream_origin = downstream[0] - translation @ upstream_axes
    return np.vstack((upstream_origin, upstream_axes))


def _pinned_bond_deviation(suite: dict[str, Any], placed: dict[tuple[int, str], np.ndarray],
                           anchor_residue_index: int,
                           anchor: dict[str, np.ndarray]) -> float:
    coordinates = dict(placed)
    coordinates.update({(anchor_residue_index, name): xyz for name, xyz in anchor.items()})
    reference = {
        (atom["residue_index"], atom["atom_name"]): np.asarray(atom["xyz_local_A"], dtype=float)
        for atom in suite["atoms"]
    }
    deviations = []
    for bond in suite["bonds"]:
        first = (bond["residue_index_1"], bond["atom_1"])
        second = (bond["residue_index_2"], bond["atom_2"])
        reference_length = float(np.linalg.norm(reference[first] - reference[second]))
        placed_length = float(np.linalg.norm(coordinates[first] - coordinates[second]))
        deviations.append(abs(placed_length - reference_length))
    return max(deviations, default=0.0)


def _baseframe_pair_metrics(library: DNAFragmentLibrary, left_suite: dict[str, Any],
                            right_suite: dict[str, Any], start_frame: np.ndarray,
                            end_frame: np.ndarray, start_anchor: dict[str, np.ndarray],
                            end_anchor: dict[str, np.ndarray]) -> dict[str, float]:
    left = library.place_suite(left_suite, start_frame, coordinate_unit="angstrom")
    right = library.place_suite(
        right_suite, left.downstream_frame, coordinate_unit="angstrom",
        downstream_frame=end_frame,
    )
    left_anchor = _suite_atom_map(left, 0)
    right_anchor = _suite_atom_map(right, 1)
    shared_left = _suite_atom_map(left, 1)
    shared_right = _suite_atom_map(right, 0)
    anchor_start = _fit_metrics(
        np.stack([left_anchor[name] for name in sorted(start_anchor)]),
        np.stack([start_anchor[name] for name in sorted(start_anchor)]),
    )
    anchor_end = _fit_metrics(
        np.stack([right_anchor[name] for name in sorted(end_anchor)]),
        np.stack([end_anchor[name] for name in sorted(end_anchor)]),
    )
    common = sorted(shared_left.keys() & shared_right.keys())
    delta = np.stack([shared_left[name] - shared_right[name] for name in common])
    shared_rmsd = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
    shared_max = float(np.linalg.norm(delta, axis=1).max())
    return {
        "start_anchor_rmsd_A": anchor_start[0],
        "start_anchor_max_displacement_A": anchor_start[1],
        "end_anchor_rmsd_A": anchor_end[0],
        "end_anchor_max_displacement_A": anchor_end[1],
        "endpoint_translation_residual_A": float(right.translation_residual_A),
        "endpoint_rotation_residual_degrees": float(right.rotation_residual_degrees),
        "shared_residue_rmsd_A": shared_rmsd,
        "shared_residue_max_displacement_A": shared_max,
    }


def _anchor_fit_pair_metrics(library: DNAFragmentLibrary, left_suite: dict[str, Any],
                             right_suite: dict[str, Any], start_frame: np.ndarray,
                             end_frame: np.ndarray, start_anchor: dict[str, np.ndarray],
                             end_anchor: dict[str, np.ndarray]) -> dict[str, float]:
    left_initial = library.place_suite(left_suite, start_frame, coordinate_unit="angstrom")
    left_mobile_map = _suite_atom_map(left_initial, 0)
    names_start = sorted(start_anchor)
    start_rotation, start_translation = rigid_fit(
        np.stack([left_mobile_map[name] for name in names_start]),
        np.stack([start_anchor[name] for name in names_start]),
    )
    left_xyz = _transform_placement(left_initial, start_rotation, start_translation)
    start_fit = np.stack([left_xyz[(0, name)] for name in names_start])
    start_rmsd, start_max = _fit_metrics(start_fit, np.stack([start_anchor[n] for n in names_start]))

    right_upstream_frame = _downstream_fit_frame(right_suite, end_frame)
    right_initial = library.place_suite(
        right_suite, right_upstream_frame, coordinate_unit="angstrom",
        downstream_frame=end_frame,
    )
    right_mobile_map = _suite_atom_map(right_initial, 1)
    names_end = sorted(end_anchor)
    end_rotation, end_translation = rigid_fit(
        np.stack([right_mobile_map[name] for name in names_end]),
        np.stack([end_anchor[name] for name in names_end]),
    )
    right_xyz = _transform_placement(right_initial, end_rotation, end_translation)
    end_fit = np.stack([right_xyz[(1, name)] for name in names_end])
    end_rmsd, end_max = _fit_metrics(end_fit, np.stack([end_anchor[n] for n in names_end]))

    shared_left = {name: xyz for (index, name), xyz in left_xyz.items() if index == 1}
    shared_right = {name: xyz for (index, name), xyz in right_xyz.items() if index == 0}
    common = sorted(shared_left.keys() & shared_right.keys())
    delta = np.stack([shared_left[name] - shared_right[name] for name in common])
    shared_rmsd = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
    shared_max = float(np.linalg.norm(delta, axis=1).max())
    left_mid_frame = _frame_from_atom_map(left_suite["sequence"][1], shared_left)
    right_mid_frame = _frame_from_atom_map(right_suite["sequence"][0], shared_right)
    middle_translation, middle_rotation = _frame_residual(left_mid_frame, right_mid_frame)

    start_pinned_bond = _pinned_bond_deviation(left_suite, left_xyz, 0, start_anchor)
    end_pinned_bond = _pinned_bond_deviation(right_suite, right_xyz, 1, end_anchor)
    start_link = float(np.linalg.norm(start_anchor["O3'"] - shared_left["P"]))
    end_link = float(np.linalg.norm(shared_right["O3'"] - end_anchor["P"]))
    start_link_deviation = abs(start_link - float(left_suite["observed_O3_prime_to_next_P_A"]))
    end_link_deviation = abs(end_link - float(right_suite["observed_O3_prime_to_next_P_A"]))
    limits = DEFAULT_LIMITS
    passed = (
        middle_translation <= limits["max_translation_residual_A"]
        and middle_rotation <= limits["max_rotation_residual_degrees"]
        and shared_rmsd <= limits["max_shared_residue_rmsd_A"]
        and shared_max <= limits["max_shared_atom_displacement_A"]
        and max(start_link_deviation, end_link_deviation)
            <= limits["max_anchor_connection_length_deviation_A"]
        and max(start_pinned_bond, end_pinned_bond) <= limits["max_bond_length_deviation_A"]
    )
    return {
        "start_anchor_fit_rmsd_A": start_rmsd,
        "start_anchor_fit_max_displacement_A": start_max,
        "end_anchor_fit_rmsd_A": end_rmsd,
        "end_anchor_fit_max_displacement_A": end_max,
        "shared_residue_rmsd_A": shared_rmsd,
        "shared_residue_max_displacement_A": shared_max,
        "middle_frame_translation_residual_A": middle_translation,
        "middle_frame_rotation_residual_degrees": middle_rotation,
        "start_anchor_pinned_bond_max_deviation_A": start_pinned_bond,
        "end_anchor_pinned_bond_max_deviation_A": end_pinned_bond,
        "start_anchor_connection_length_deviation_A": start_link_deviation,
        "end_anchor_connection_length_deviation_A": end_link_deviation,
        "passes_existing_geometry_limits": bool(passed),
    }


def diagnose(template_library_paths: tuple[Path, ...] = TEMPLATE_LIBRARIES,
             target_library_path: Path = TARGET_LIBRARY) -> dict[str, Any]:
    import gemmi

    libraries = [DNAFragmentLibrary.from_json(path) for path in template_library_paths]
    suites = [suite for library in libraries for suite in library._suites]
    source_files = tuple(source for library in libraries for source in library.source_files)
    library = DNAFragmentLibrary(suites, source_files=source_files)
    target_data = json.loads(target_library_path.read_text(encoding="utf-8"))
    target_path = PROJECT_ROOT / target_data["source_files"][0]["file"]
    structure = gemmi.read_structure(str(target_path))
    if len(structure) != 1:
        raise ValueError("Expected one target coordinate model")
    target_suites = {suite["suite_id"]: suite for suite in json.loads(
        target_library_path.read_text(encoding="utf-8"))["suites"]}

    rows = []
    for suite_ids in PANEL_TARGET_SUITE_IDS:
        first, second = (target_suites[suite_id] for suite_id in suite_ids)
        chain_id = first["chain_id"]
        chain = next(item for item in structure[0] if item.name == chain_id)
        residues = {str(residue.seqid.num): residue for residue in chain
                    if residue.name in RESIDUE_TO_BASE}
        sequence = first["sequence"] + second["sequence"][1]
        start_residue = residues[first["source_residues"][0]]
        end_residue = residues[second["source_residues"][1]]
        start_frame = _frame_for_residue(start_residue, sequence[0])
        end_frame = _frame_for_residue(end_residue, sequence[2])
        start_anchor = {k: np.asarray(v, dtype=float) for k, v in _anchor_atoms(start_residue).items()}
        end_anchor = {k: np.asarray(v, dtype=float) for k, v in _anchor_atoms(end_residue).items()}
        left_suites = library.get_suites(sequence[:2])
        right_suites = library.get_suites(sequence[1:])
        pairs = []
        for left_suite in left_suites:
            for right_suite in right_suites:
                base = _baseframe_pair_metrics(
                    library, left_suite, right_suite, start_frame, end_frame,
                    start_anchor, end_anchor,
                )
                fitted = _anchor_fit_pair_metrics(
                    library, left_suite, right_suite, start_frame, end_frame,
                    start_anchor, end_anchor,
                )
                pairs.append({
                    "suite_ids": [left_suite["suite_id"], right_suite["suite_id"]],
                    "baseframe": base,
                    "full_anchor_rigid_fit": fitted,
                })
        survivors = [item for item in pairs
                     if item["full_anchor_rigid_fit"]["passes_existing_geometry_limits"]]
        gate_order = (
            "middle_frame_geometry",
            "shared_residue_geometry",
            "pinned_bond_geometry",
            "anchor_connection_geometry",
        )
        gate_failure_counts = {name: 0 for name in gate_order}
        first_failure_counts = {name: 0 for name in gate_order}
        for item in pairs:
            fit = item["full_anchor_rigid_fit"]
            failed = []
            if (fit["middle_frame_translation_residual_A"]
                    > DEFAULT_LIMITS["max_translation_residual_A"]
                    or fit["middle_frame_rotation_residual_degrees"]
                    > DEFAULT_LIMITS["max_rotation_residual_degrees"]):
                failed.append("middle_frame_geometry")
            if (fit["shared_residue_rmsd_A"] > DEFAULT_LIMITS["max_shared_residue_rmsd_A"]
                    or fit["shared_residue_max_displacement_A"]
                    > DEFAULT_LIMITS["max_shared_atom_displacement_A"]):
                failed.append("shared_residue_geometry")
            if (max(fit["start_anchor_pinned_bond_max_deviation_A"],
                    fit["end_anchor_pinned_bond_max_deviation_A"])
                    > DEFAULT_LIMITS["max_bond_length_deviation_A"]):
                failed.append("pinned_bond_geometry")
            if (max(fit["start_anchor_connection_length_deviation_A"],
                    fit["end_anchor_connection_length_deviation_A"])
                    > DEFAULT_LIMITS["max_anchor_connection_length_deviation_A"]):
                failed.append("anchor_connection_geometry")
            for reason in failed:
                gate_failure_counts[reason] += 1
            if failed:
                first_failure_counts[failed[0]] += 1
        base_anchor_max = [
            max(item["baseframe"]["start_anchor_max_displacement_A"],
                item["baseframe"]["end_anchor_max_displacement_A"])
            for item in pairs
        ]
        fitted_anchor_max = [
            max(item["full_anchor_rigid_fit"]["start_anchor_fit_max_displacement_A"],
                item["full_anchor_rigid_fit"]["end_anchor_fit_max_displacement_A"])
            for item in pairs
        ]
        ranking = sorted(pairs, key=lambda item: (
            item["full_anchor_rigid_fit"]["middle_frame_translation_residual_A"]
                / DEFAULT_LIMITS["max_translation_residual_A"]
            + item["full_anchor_rigid_fit"]["middle_frame_rotation_residual_degrees"]
                / DEFAULT_LIMITS["max_rotation_residual_degrees"]
            + item["full_anchor_rigid_fit"]["shared_residue_rmsd_A"]
                / DEFAULT_LIMITS["max_shared_residue_rmsd_A"],
            item["suite_ids"],
        ))
        rows.append({
            "target_suite_ids": list(suite_ids),
            "target_sequence_including_anchors": sequence,
            "target_chain": chain_id,
            "left_suite_count": len(left_suites),
            "right_suite_count": len(right_suites),
            "pair_count": len(pairs),
            "full_anchor_rigid_fit_survivor_count": len(survivors),
            "gate_failure_counts": gate_failure_counts,
            "first_failure_counts": first_failure_counts,
            "max_anchor_displacement_A": {
                "baseframe_placement": _summary(base_anchor_max),
                "full_anchor_rigid_fit_residual": _summary(fitted_anchor_max),
            },
            "best_full_anchor_rigid_fit_pairs": ranking[:5],
        })

    return {
        "schema_version": 1,
        "protocol": "masked-dna-gap-baseframe-vs-full-anchor-kabsch-diagnostic-v1",
        "template_library_files": [path.resolve().relative_to(PROJECT_ROOT).as_posix()
                                    for path in template_library_paths],
        "template_entry_ids": sorted({suite["source_entry"] for suite in suites}),
        "template_suite_count": len(suites),
        "target_entry": "1KX5",
        "window_count": len(rows),
        "masked_middle_coordinates_used": False,
        "formal_thresholds_changed": False,
        "scientific_acceptance": False,
        "environment_screening": "not_evaluated",
        "interpretation_limit": (
            "Full-anchor Kabsch is a rigid-fit diagnostic. Candidates are screened using existing geometric limits "
            "for middle-frame mismatch, shared-residue consistency, pinned covalent bond deviation, and anchor-link "
            "length deviation. It does not optimize internal coordinates, assess nonbonded environment, or establish "
            "experimental accuracy or independent-family generalization."
        ),
        "runs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = diagnose()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "windows": report["window_count"],
        "pairs": sum(row["pair_count"] for row in report["runs"]),
        "survivors": sum(row["full_anchor_rigid_fit_survivor_count"] for row in report["runs"]),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
