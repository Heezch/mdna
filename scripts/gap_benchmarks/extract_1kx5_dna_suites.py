"""Extract source-traceable internal DNA dinucleotide suites from a PDB.

Each output covers one coordinate entry. Use a separate entry ID when the
input filename does not match its PDB ID. A one-entry library is not an
independent benchmark or a general DNA conformer library.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from mdna.chemistry import get_nucleotide_component

DEFAULT_INPUT = PROJECT_ROOT / "docs/notebooks/pdbs/1kx5.pdb"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/gap_benchmarks/1kx5_dna_suites.json"
RESIDUE_TO_BASE = {"DA": "A", "DC": "C", "DG": "G", "DT": "T"}
BASE_TO_COMPONENT = {"A": "A", "C": "C", "G": "G", "T": "T"}
PURINE_RING = ("N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4")
PYRIMIDINE_RING = ("N1", "C2", "N3", "C4", "C5", "C6")
BASE_RING = {"A": PURINE_RING, "G": PURINE_RING, "C": PYRIMIDINE_RING, "T": PYRIMIDINE_RING}
FRAME_ANCHORS = {"A": ("N9", "C4"), "G": ("N9", "C4"), "C": ("N1", "C2"), "T": ("N1", "C2")}
DINUCLEOTIDES = tuple("".join(pair) for pair in itertools.product("ACGT", repeat=2))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atom_coordinates(residue: Any) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Select one alternate per atom by occupancy, preferring blank then A."""
    selected: dict[str, Any] = {}
    for atom in residue:
        name = atom.name.strip()
        previous = selected.get(name)
        rank = (float(atom.occ), atom.altloc in ("\x00", " "), atom.altloc == "A")
        previous_rank = (
            (float(previous.occ), previous.altloc in ("\x00", " "), previous.altloc == "A")
            if previous is not None else None
        )
        if previous is None or rank > previous_rank:
            selected[name] = atom
    coordinates = {}
    elements = {}
    for name, atom in selected.items():
        point = np.asarray([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)
        if not np.isfinite(point).all():
            raise ValueError(f"{residue.name}:{residue.seqid} atom {name} has non-finite coordinates")
        coordinates[name] = point
        elements[name] = atom.element.name.upper()
    return coordinates, elements


def make_base_frame(base: str, coordinates: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Build an angstrom frame matching MDNA ReferenceBase axis order."""
    if base not in BASE_RING:
        raise ValueError(f"Unsupported DNA base {base!r}.")
    required = set(BASE_RING[base]) | {"C1'"}
    missing = required - coordinates.keys()
    if missing:
        raise ValueError(f"DNA {base} residue lacks base-frame atoms: {sorted(missing)}")
    glycosidic, ring_carbon = FRAME_ANCHORS[base]
    n_atom, c1_atom, c_atom = coordinates[glycosidic], coordinates["C1'"], coordinates[ring_carbon]
    z_axis = np.cross(n_atom - c1_atom, n_atom - c_atom)
    z_norm = np.linalg.norm(z_axis)
    glycosidic_vector = c1_atom - n_atom
    glycosidic_norm = np.linalg.norm(glycosidic_vector)
    if min(z_norm, glycosidic_norm) <= np.finfo(float).eps:
        raise ValueError(f"DNA {base} base-frame anchors are degenerate")
    z_axis /= z_norm
    glycosidic_vector /= glycosidic_norm

    def rotate(vector: np.ndarray, degrees: float) -> np.ndarray:
        angle = np.deg2rad(degrees)
        return (vector * np.cos(angle) + np.cross(z_axis, vector) * np.sin(angle)
                + z_axis * np.dot(z_axis, vector) * (1 - np.cos(angle)))

    origin = n_atom + rotate(glycosidic_vector, 141.478) * 4.702
    long_axis = rotate(glycosidic_vector, -54.418)
    short_axis = np.cross(long_axis, z_axis)
    short_axis /= np.linalg.norm(short_axis)
    frame = np.column_stack((short_axis, long_axis, z_axis))
    if not np.isfinite(frame).all() or not np.allclose(frame.T @ frame, np.eye(3), atol=1e-6) or np.linalg.det(frame) < 1 - 1e-6:
        raise ValueError(f"DNA {base} frame is not a right-handed orthonormal frame")
    return origin, frame


def extract_suite(entry_id: str, chain_id: str, upstream: Any, downstream: Any,
                  source_hash: str, chain_sequence: str) -> dict[str, Any]:
    first_base, second_base = RESIDUE_TO_BASE[upstream.name], RESIDUE_TO_BASE[downstream.name]
    first_coords, first_elements = atom_coordinates(upstream)
    second_coords, second_elements = atom_coordinates(downstream)
    residue_data = ((first_base, first_coords, first_elements), (second_base, second_coords, second_elements))
    components = []
    for base, coordinates, elements in residue_data:
        component = get_nucleotide_component(BASE_TO_COMPONENT[base])
        expected = {atom.name: atom.element.upper() for atom in component.atoms if atom.element.upper() != "H"}
        observed = {name: element for name, element in elements.items() if element != "H"}
        if expected != observed:
            raise ValueError(f"{entry_id}:{chain_id}:{base} atom mismatch; missing={sorted(expected.keys()-observed.keys())}, unexpected={sorted(observed.keys()-expected.keys())}")
        components.append((component, expected))

    first_origin, first_frame = make_base_frame(first_base, first_coords)
    second_origin, second_frame = make_base_frame(second_base, second_coords)
    atoms = []
    bonds = []
    for residue_index, ((base, coordinates, elements), (component, expected)) in enumerate(zip(residue_data, components)):
        for name in sorted(expected):
            local = (coordinates[name] - first_origin) @ first_frame
            atoms.append({"residue_index": residue_index, "atom_name": name,
                          "element": elements[name], "xyz_local_A": [float(value) for value in local]})
        bonds.extend({"residue_index_1": residue_index, "atom_1": bond.atom_1,
                      "residue_index_2": residue_index, "atom_2": bond.atom_2,
                      "order": bond.order}
                     for bond in component.bonds
                     if bond.atom_1 in expected and bond.atom_2 in expected)
    bonds.append({"residue_index_1": 0, "atom_1": "O3'", "residue_index_2": 1,
                  "atom_2": "P", "order": "single"})
    link_distance = float(np.linalg.norm(first_coords["O3'"] - second_coords["P"]))
    relative_rotation = first_frame.T @ second_frame
    return {
        "suite_id": f"{entry_id.upper()}:{chain_id}:{upstream.seqid}-{downstream.seqid}",
        "sequence": first_base + second_base,
        "source_entry": entry_id.upper(), "source_coordinate_sha256": source_hash,
        "chain_id": chain_id, "source_chain_sequence": chain_sequence,
        "source_residues": [str(upstream.seqid), str(downstream.seqid)],
        "component_ids": [components[0][0].component_id, components[1][0].component_id],
        "component_sha256": [components[0][0].sha256, components[1][0].sha256],
        "coordinate_unit": "angstrom",
        "coordinate_frame": "MDNA ReferenceBase single-base frame with [b_D,b_L,b_N] axes; not a base-pair frame",
        "downstream_base_frame_relative_to_upstream": {
            "translation_A": [float(value) for value in (second_origin - first_origin) @ first_frame],
            "rotation_matrix": relative_rotation.tolist()},
        "observed_O3_prime_to_next_P_A": link_distance,
        "atoms": atoms, "bonds": bonds,
    }


def validate_suite(suite: dict[str, Any]) -> None:
    keys = {(atom["residue_index"], atom["atom_name"]) for atom in suite["atoms"]}
    if len(keys) != len(suite["atoms"]):
        raise ValueError(f"{suite['suite_id']}: duplicate atom records")
    if any(not np.isfinite(np.asarray(atom["xyz_local_A"], dtype=float)).all() for atom in suite["atoms"]):
        raise ValueError(f"{suite['suite_id']}: non-finite coordinates")
    for bond in suite["bonds"]:
        if (bond["residue_index_1"], bond["atom_1"]) not in keys or (bond["residue_index_2"], bond["atom_2"]) not in keys:
            raise ValueError(f"{suite['suite_id']}: bond endpoint is absent")
    transform = suite["downstream_base_frame_relative_to_upstream"]
    rotation = np.asarray(transform["rotation_matrix"], dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) or np.linalg.det(rotation) < 1 - 1e-7:
        raise ValueError(f"{suite['suite_id']}: relative transform is not a proper rigid rotation")
    if not np.isfinite(suite["observed_O3_prime_to_next_P_A"]):
        raise ValueError(f"{suite['suite_id']}: non-finite O3'-P distance")


def build_library(input_path: Path, entry_id: str | None = None) -> dict[str, Any]:
    import gemmi

    entry_id = (entry_id or input_path.stem).strip().upper()
    if not entry_id or any(character.isspace() for character in entry_id):
        raise ValueError("entry_id must be a non-empty token without whitespace")
    source_hash = sha256(input_path)
    structure = gemmi.read_structure(str(input_path))
    if len(structure) != 1:
        raise ValueError(f"Expected one coordinate model; found {len(structure)}")
    suites = []
    for chain in structure[0]:
        standard = [(index, residue) for index, residue in enumerate(chain) if residue.name in RESIDUE_TO_BASE]
        sequence = "".join(RESIDUE_TO_BASE[residue.name] for _, residue in standard)
        for pair_position, ((i, first), (j, second)) in enumerate(zip(standard, standard[1:])):
            # End residues can have hydroxyl/phosphate states unlike an
            # internal phosphodiester suite; omit both chain-terminal pairs.
            if pair_position == 0 or pair_position >= len(standard) - 2:
                continue
            if j != i + 1:
                continue
            first_coords, _ = atom_coordinates(first)
            second_coords, _ = atom_coordinates(second)
            if "O3'" not in first_coords or "P" not in second_coords:
                continue
            suite = extract_suite(entry_id, chain.name, first, second, source_hash, sequence)
            validate_suite(suite)
            suites.append(suite)
    if not suites:
        raise ValueError("No complete adjacent standard DNA suites were extracted")
    counts = Counter(suite["sequence"] for suite in suites)
    return {
        "schema_version": 1,
        "library_type": "experimental internal DNA dinucleotide suites",
        "coordinate_unit": "angstrom",
        "source_files": [{"file": input_path.resolve().relative_to(PROJECT_ROOT).as_posix(), "sha256": source_hash}],
        "chemistry_registry": "mdna/chemistry.py:NUCLEOTIDE_REGISTRY",
        "component_source_policy": "DNA atom/bond definitions are adapted from bundled HDF5 templates; each suite stores the component template SHA-256.",
        "suite_count": len(suites),
        "sequence_type_count": len(counts),
        "suite_counts_by_sequence": {key: counts.get(key, 0) for key in DINUCLEOTIDES},
        "unobserved_sequence_dinucleotides": sorted(set(DINUCLEOTIDES) - counts.keys()),
        "sources_represent_independent_families": False,
        "interpretation_limit": f"All suites come from one structure entry ({entry_id}); chains within the entry are not independent families. This source set is not an independent benchmark or general DNA conformer library.",
        "suites": suites,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--entry-id", type=str, default=None,
                        help="source PDB entry identifier; defaults to the input filename stem")
    args = parser.parse_args()
    report = build_library(args.input, args.entry_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("suite_count", "sequence_type_count", "suite_counts_by_sequence", "unobserved_sequence_dinucleotides")}, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
