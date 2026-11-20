"""Evaluate terminal-suite transfer with one RNA entry held out at a time.

The held-out base-frame poses are used as explicit path constraints. The
held-out suite atom coordinates are used only for scoring candidate transfers.
All six structures share one sequence/family, so this is conditional transfer
evidence, not a general RNA validation benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rna_pilot.extract_rna_suite_library import (  # noqa: E402
    _atom_coordinates,
    make_base_frame,
)


def _angle_degrees(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _rmsd(target: dict[str, np.ndarray], placed: dict[str, np.ndarray], names: set[str]) -> float:
    shared = sorted(names & target.keys() & placed.keys())
    if not shared:
        raise ValueError("No shared atom coordinates are available for RMSD.")
    delta = np.stack([target[name] - placed[name] for name in shared])
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def _held_out_suite(structure: Any, chain_id: str, role: str) -> tuple[Any, Any]:
    chain = next((item for item in structure[0] if item.name == chain_id), None)
    if chain is None:
        raise ValueError(f"Held-out entry lacks chain {chain_id!r}.")
    residues = [residue for residue in chain if residue.name.strip().upper() in "ACGU"]
    if role == "five_prime":
        return residues[0], residues[1]
    if role == "three_prime":
        return residues[-2], residues[-1]
    raise ValueError(f"Unsupported terminal role {role!r}.")


def evaluate_loo(library: dict[str, Any], reference_dir: Path) -> dict[str, Any]:
    import gemmi

    candidates_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for candidate in library["suites"]:
        key = (candidate["terminal_role"], candidate["sequence"])
        candidates_by_key.setdefault(key, []).append(candidate)

    cases = []
    source_hashes = []
    backbone_names = {
        "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'",
        "O3'", "C2'", "O2'", "C1'",
    }
    sugar_names = {"C1'", "C2'", "C3'", "C4'", "C5'", "O2'", "O3'", "O4'", "O5'"}

    for source in library["sources"]:
        entry_id = Path(source["file"]).stem.upper()
        path = reference_dir / source["file"]
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != source["sha256"]:
            raise ValueError(f"{path.name}: hash differs from the terminal library manifest")
        source_hashes.append({"file": path.name, "sha256": actual_hash})
        structure = gemmi.read_structure(str(path))
        for terminal in (item for item in library["suites"] if item["source_entry"] == entry_id):
            role = terminal["terminal_role"]
            chain_id = terminal["chain_id"]
            upstream, downstream = _held_out_suite(structure, chain_id, role)
            upstream_base = upstream.name.strip().upper()
            downstream_base = downstream.name.strip().upper()
            sequence = upstream_base + downstream_base
            if sequence != terminal["sequence"]:
                raise ValueError(f"{entry_id}:{chain_id}:{role}: held-out sequence mismatch")

            upstream_coordinates, upstream_elements = _atom_coordinates(upstream)
            downstream_coordinates, downstream_elements = _atom_coordinates(downstream)
            upstream_origin, upstream_frame = make_base_frame(upstream_base, upstream_coordinates)
            target_down_origin, target_down_frame = make_base_frame(
                downstream_base, downstream_coordinates
            )
            target_coordinates = {}
            target_elements = {}
            for residue_index, coordinates, elements in (
                (0, upstream_coordinates, upstream_elements),
                (1, downstream_coordinates, downstream_elements),
            ):
                for atom_name, coordinate in coordinates.items():
                    if elements[atom_name] in {"H", "D"}:
                        continue
                    target_coordinates[(residue_index, atom_name)] = coordinate
                    target_elements[(residue_index, atom_name)] = elements[atom_name]

            candidate_records = candidates_by_key[(role, sequence)]
            candidate_results = []
            for candidate in candidate_records:
                if candidate["source_entry"] == entry_id:
                    continue
                relative = candidate["downstream_base_frame_relative_to_upstream"]
                relative_translation = np.asarray(relative["translation_A"], dtype=float)
                relative_rotation = np.asarray(relative["rotation_matrix"], dtype=float)
                placed_atoms = {}
                placed_elements = {}
                for atom in candidate["atoms"]:
                    local = np.asarray(atom["xyz_local_A"], dtype=float)
                    key = (atom["residue_index"], atom["atom_name"])
                    placed_atoms[key] = local @ upstream_frame.T + upstream_origin
                    placed_elements[key] = atom["element"].upper()

                predicted_down_origin = (
                    relative_translation @ upstream_frame.T + upstream_origin
                )
                predicted_down_frame = upstream_frame @ relative_rotation
                frame_translation_residual = float(
                    np.linalg.norm(predicted_down_origin - target_down_origin)
                )
                frame_rotation_residual = _angle_degrees(
                    predicted_down_frame.T @ target_down_frame
                )

                all_keys = set(target_coordinates) & set(placed_atoms)
                target_by_name = {f"{res}:{name}": xyz for (res, name), xyz in target_coordinates.items()}
                placed_by_name = {f"{res}:{name}": xyz for (res, name), xyz in placed_atoms.items()}
                all_names = {f"{res}:{name}" for res, name in all_keys}
                backbone = {f"{res}:{name}" for res, name in all_keys if name in backbone_names}
                sugar = {f"{res}:{name}" for res, name in all_keys if name in sugar_names}
                candidate_results.append({
                    "candidate_suite_id": candidate["suite_id"],
                    "candidate_source_entry": candidate["source_entry"],
                    "shared_heavy_atom_count": len(all_keys),
                    "shared_atom_elements_match": all(
                        target_elements[key] == placed_elements[key] for key in all_keys
                    ),
                    "downstream_frame_translation_residual_A": frame_translation_residual,
                    "downstream_frame_rotation_residual_degrees": frame_rotation_residual,
                    "all_shared_heavy_atom_rmsd_A": _rmsd(target_by_name, placed_by_name, all_names),
                    "sugar_rmsd_A": _rmsd(target_by_name, placed_by_name, sugar),
                    "backbone_rmsd_A": _rmsd(target_by_name, placed_by_name, backbone),
                })

            if not candidate_results:
                raise ValueError(f"{entry_id}:{chain_id}:{role}: no leave-one-entry-out candidates")
            cases.append({
                "held_out_entry": entry_id,
                "held_out_chain": chain_id,
                "terminal_role": role,
                "sequence": sequence,
                "held_out_source_suite_id": terminal["suite_id"],
                "candidate_count": len(candidate_results),
                "candidate_selection": "none; all candidates reported without held-out atom-based ranking",
                "candidates": candidate_results,
            })

    flat = [candidate for case in cases for candidate in case["candidates"]]
    metric_names = (
        "downstream_frame_translation_residual_A",
        "downstream_frame_rotation_residual_degrees",
        "all_shared_heavy_atom_rmsd_A",
        "sugar_rmsd_A",
        "backbone_rmsd_A",
    )
    summary = {
        role: {
            metric: {
                "n": int(len(values)),
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "maximum": float(np.max(values)),
            }
            for metric in metric_names
            for values in [[
                candidate[metric]
                for case in cases if case["terminal_role"] == role
                for candidate in case["candidates"]
            ]]
        }
        for role in ("five_prime", "three_prime")
    }
    return {
        "method": (
            "Leave one PDB entry out. Place every same-role, same-sequence candidate from "
            "other entries into the held-out upstream base frame. Compare its predicted "
            "downstream frame and atom coordinates with the held-out reference."
        ),
        "candidate_selection": (
            "No candidate is selected or ranked by held-out atom coordinates. Metrics are "
            "reported for every cross-entry candidate; target downstream frames are explicit "
            "oracle pose constraints from the held-out structures."
        ),
        "source_entries": source_hashes,
        "case_count": len(cases),
        "candidate_pair_count": len(flat),
        "summary_by_terminal_role": summary,
        "cases": cases,
        "interpretation_limit": (
            "All inputs are the same 8-nt sequence and related duplex crystal structures. "
            "This tests frame-conditioned transfer within this structure set, not blind "
            "sequence generalization, natural RNA loops, a gap solver, or a full-chain builder."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library",
        type=Path,
        default=PROJECT_ROOT / "artifacts/rna_pilot/reference/rna_terminal_suite_library.json",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/rna_pilot/reference/ion_series",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/rna_pilot/reference/terminal_suite_loo_evaluation.json",
    )
    args = parser.parse_args()
    library = json.loads(args.library.read_text(encoding="utf-8"))
    report = evaluate_loo(library, args.reference_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "case_count": report["case_count"],
        "candidate_pair_count": report["candidate_pair_count"],
        "summary_by_terminal_role": report["summary_by_terminal_role"],
    }, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
