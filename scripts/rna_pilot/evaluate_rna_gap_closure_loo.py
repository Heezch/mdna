"""Evaluate a one-residue RNA gap with one PDB entry excluded from the library.

The solver receives only the two observed anchor frames and atom coordinates.
Coordinates of the masked residue are retained by the evaluator and compared
only after candidate generation. This entry is one member of a shared-sequence
ion series, so the result is a source-held-out software/geometry check, not an
independent RNA-family benchmark.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import gemmi
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mdna import close_rna_gap_from_anchors
from mdna.rna_fragments import RNAFragmentLibrary
from scripts.rna_pilot.extract_rna_suite_library import _atom_coordinates, make_base_frame


REFERENCE_DIR = PROJECT_ROOT / "artifacts/rna_pilot/reference"
TARGET_ENTRY = "4U3R"
TARGET_CHAIN = "A"
ANCHOR_START = 1
ANCHOR_END = 3
MIN_ENVIRONMENT_DISTANCE_A = 1.5


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_frame(residue):
    atoms, _ = _atom_coordinates(residue)
    origin, axes_columns = make_base_frame(residue.name, atoms)
    return np.vstack((origin, axes_columns.T))


def _anchor_heavy_atoms(residue):
    atoms = {}
    for atom in residue:
        if atom.element.name.upper() in {"H", "D"}:
            continue
        atoms[atom.name] = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)
    return atoms


def _environment_heavy_atoms(structure, excluded_chain_name):
    atoms = []
    for chain in structure[0]:
        if chain.name == excluded_chain_name:
            continue
        for residue in chain:
            for atom in residue:
                element = atom.element.name.upper()
                if element in {"H", "D"}:
                    continue
                altloc = str(atom.altloc).strip() or "."
                atom_id = (
                    f"{chain.name}:{residue.seqid.num}:{residue.name}:"
                    f"{atom.name}:{altloc}"
                )
                atoms.append({
                    "atom_id": atom_id,
                    "element": element,
                    "xyz_A": [atom.pos.x, atom.pos.y, atom.pos.z],
                })
    return atoms


def _xyz_by_residue(assembly):
    return {
        (atom["residue_index"], atom["atom_name"]): np.asarray(atom["xyz_A"], dtype=float)
        for atom in assembly.atoms
    }


def _rmsd(reference, candidate, atom_names):
    common = sorted(atom_names & reference.keys() & candidate.keys())
    if not common:
        return None, 0
    differences = np.stack([candidate[name] - reference[name] for name in common])
    return float(np.sqrt(np.mean(np.sum(differences * differences, axis=1)))), len(common)


def evaluate() -> dict:
    coordinate_path = REFERENCE_DIR / "ion_series" / "4u3r.cif"
    library_path = REFERENCE_DIR / "rna_suite_library.json"
    coordinate_sha256 = hashlib.sha256(coordinate_path.read_bytes()).hexdigest()

    structure = gemmi.read_structure(str(coordinate_path))
    if not structure:
        raise ValueError(f"No coordinate model in {coordinate_path}.")
    chain = next((item for item in structure[0] if item.name == TARGET_CHAIN), None)
    if chain is None:
        raise ValueError(f"Target chain {TARGET_CHAIN!r} is absent from {TARGET_ENTRY}.")
    residues = [residue for residue in chain if residue.name in "ACGU"]
    sequence = "".join(residue.name for residue in residues)
    if not (0 <= ANCHOR_START < ANCHOR_END < len(residues)):
        raise ValueError("Configured RNA gap anchor positions are outside the target chain.")
    if ANCHOR_END - ANCHOR_START != 2:
        raise ValueError("This pilot evaluates exactly one masked residue between two anchors.")
    gap_sequence = sequence[ANCHOR_START:ANCHOR_END + 1]

    with library_path.open("r", encoding="utf-8") as handle:
        library_data = json.load(handle)
    source_records = library_data["sources"]
    allowed_sources = [
        source for source in source_records
        if Path(source["file"]).stem.upper() != TARGET_ENTRY
    ]
    heldout_sources = [
        source for source in source_records
        if Path(source["file"]).stem.upper() == TARGET_ENTRY
    ]
    suites = [
        suite for suite in library_data["suites"]
        if not suite["suite_id"].upper().startswith(f"{TARGET_ENTRY}:")
    ]
    if not heldout_sources or not allowed_sources:
        raise ValueError("The source-level holdout split could not be verified.")
    source_manifest_payload = json.dumps(
        allowed_sources, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    library = RNAFragmentLibrary(
        suites,
        source_manifest_sha256=_sha256_bytes(source_manifest_payload),
        chemistry_registry_sha256=library_data["chemistry_registry_sha256"],
    )

    start_residue = residues[ANCHOR_START]
    end_residue = residues[ANCHOR_END]
    start_frame = _make_frame(start_residue)
    end_frame = _make_frame(end_residue)
    start_anchor_atoms = _anchor_heavy_atoms(start_residue)
    end_anchor_atoms = _anchor_heavy_atoms(end_residue)
    environment_atoms = _environment_heavy_atoms(structure, TARGET_CHAIN)

    # The target structure is in evaluator memory, but the solver call below
    # receives only observed anchor data. The masked residue is read again for
    # scoring after the candidate search has returned.
    closure = close_rna_gap_from_anchors(
        library,
        gap_sequence,
        start_frame,
        end_frame,
        start_anchor_atoms,
        end_anchor_atoms,
        coordinate_unit="angstrom",
        max_translation_residual_A=1.0,
        max_rotation_residual_degrees=20.0,
        max_shared_residue_rmsd_A=1.5,
        max_shared_atom_displacement_A=1.0,
        max_bond_length_deviation_A=0.25,
        max_anchor_atom_displacement_A=1.0,
        max_anchor_connection_length_deviation_A=0.25,
        max_search_states=100_000,
        max_returned_candidates=25,
        environment_atoms=environment_atoms,
        min_environment_distance_A=MIN_ENVIRONMENT_DISTANCE_A,
    )

    truth_residue = residues[ANCHOR_START + 1]
    truth_coordinates, truth_elements = _atom_coordinates(truth_residue)
    truth_heavy = {
        name: xyz for name, xyz in truth_coordinates.items()
        if truth_elements[name].upper() not in {"H", "D"}
    }
    backbone_names = {
        "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'",
        "O3'", "C2'", "O2'", "C1'",
    }
    scored_candidates = []
    target_o3_to_p = []
    for index in range(ANCHOR_START, ANCHOR_END):
        first_atoms, _ = _atom_coordinates(residues[index])
        second_atoms, _ = _atom_coordinates(residues[index + 1])
        if "O3'" in first_atoms and "P" in second_atoms:
            target_o3_to_p.append(float(np.linalg.norm(first_atoms["O3'"] - second_atoms["P"])))
    for rank, candidate in enumerate(closure.candidates, start=1):
        model_by_key = _xyz_by_residue(candidate.assembly)
        model_missing = {
            name: xyz for (residue_index, name), xyz in model_by_key.items()
            if residue_index == 1
        }
        all_atom_rmsd, common_atom_count = _rmsd(
            truth_heavy, model_missing, set(truth_heavy)
        )
        backbone_rmsd, common_backbone_count = _rmsd(
            {name: xyz for name, xyz in truth_heavy.items() if name in backbone_names},
            {name: xyz for name, xyz in model_missing.items() if name in backbone_names},
            backbone_names,
        )
        candidate_o3_to_p = []
        for residue_index in range(len(gap_sequence) - 1):
            o3 = model_by_key.get((residue_index, "O3'"))
            phosphate = model_by_key.get((residue_index + 1, "P"))
            if o3 is not None and phosphate is not None:
                candidate_o3_to_p.append(float(np.linalg.norm(o3 - phosphate)))
        scored_candidates.append({
            "rank": rank,
            **candidate.to_report(),
            "masked_residue_heavy_atom_rmsd_A": all_atom_rmsd,
            "masked_residue_common_heavy_atom_count": common_atom_count,
            "masked_residue_truth_heavy_atom_count": len(truth_heavy),
            "masked_residue_common_atom_coverage": (
                common_atom_count / len(truth_heavy) if truth_heavy else None
            ),
            "masked_residue_backbone_rmsd_A": backbone_rmsd,
            "masked_residue_common_backbone_atom_count": common_backbone_count,
            "inter_residue_O3_prime_P_distances_A": candidate_o3_to_p,
        })

    return {
        "schema_version": 1,
        "task": "one-residue RNA gap; entry-level suite holdout",
        "target_entry": TARGET_ENTRY,
        "target_chain": TARGET_CHAIN,
        "target_coordinate_sha256": coordinate_sha256,
        "target_sequence": sequence,
        "gap_sequence_including_anchors": gap_sequence,
        "anchor_residue_positions_zero_based": [ANCHOR_START, ANCHOR_END],
        "masked_residue_position_zero_based": ANCHOR_START + 1,
        "heldout_source_entries": [Path(source["file"]).stem.upper() for source in heldout_sources],
        "allowed_library_source_entries": [Path(source["file"]).stem.upper() for source in allowed_sources],
        "allowed_suite_count": len(suites),
        "environment_screening": {
            "atom_count": len(environment_atoms),
            "scope": "heavy atoms in non-target chains; target chain excluded",
            "minimum_distance_cutoff_A": MIN_ENVIRONMENT_DISTANCE_A,
            "interpretation": (
                "Uniform severe-overlap screen only; not element-specific van der Waals, "
                "force-field energy, or full nonbonded validation."
            ),
        },
        "closure": closure.to_report(),
        "scored_candidates": scored_candidates,
        "target_inter_residue_O3_prime_P_distances_A": target_o3_to_p,
        "geometry_evidence_scope": (
            "Anchor-conditioned heavy-atom comparison for one residue; target entry excluded "
            "from suite candidates. The six source entries share one RNA sequence and structural "
            "family, so this is not an independent-family blind benchmark."
        ),
        "scientific_acceptance": False,
    }


def main() -> None:
    report = evaluate()
    output_path = (
        PROJECT_ROOT
        / "artifacts/rna_pilot/gap_benchmarks/4u3r_A_CGU_single_residue.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output_path),
        "closure_status": report["closure"]["status"],
        "search_complete": report["closure"]["search_complete"],
        "expanded_states": report["closure"]["expanded_states"],
        "candidate_count": len(report["scored_candidates"]),
        "best_masked_residue_heavy_atom_rmsd_A": (
            report["scored_candidates"][0]["masked_residue_heavy_atom_rmsd_A"]
            if report["scored_candidates"] else None
        ),
        "scientific_acceptance": report["scientific_acceptance"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
