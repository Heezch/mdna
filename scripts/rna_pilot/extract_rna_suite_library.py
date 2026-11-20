"""Extract source-traceable internal RNA dinucleotide fragments.

Each suite contains two consecutive internal residues, heavy-atom coordinates
from an experimental polymer, the corresponding CCD-derived intracomponent
bonds, their O3'-P inter-residue bond, and the downstream base frame relative
to the upstream base frame. Terminal suites are excluded because their
phosphate/hydroxyl atom states differ from internal residues.

Run in the project WSL ``dip2`` environment::

    python scripts/rna_pilot/extract_rna_suite_library.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


ENTRY_IDS = ("4u3l", "4u3o", "4u3p", "4u3r", "4u47", "4u78")
PURINE_RING = ("N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4")
PYRIMIDINE_RING = ("N1", "C2", "N3", "C4", "C5", "C6")
BASE_RING = {"A": PURINE_RING, "G": PURINE_RING, "C": PYRIMIDINE_RING, "U": PYRIMIDINE_RING}
FRAME_ANCHORS = {"A": ("N9", "C4"), "G": ("N9", "C4"), "C": ("N1", "C2"), "U": ("N1", "C2")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_base_frame(
    base: str,
    atom_coordinates: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return a right-handed base frame as (origin, column-axis matrix).

    The frame follows the Tsukuba single-base convention used by
    ``mdna.geometry.ReferenceBase``. Coordinates are supplied in angstroms,
    so its 0.4702 nm reference-point displacement is converted to 4.702 Å.
    The output basis is right-handed. These are local base frames, not
    base-pair or X3DNA frames.
    """

    code = base.upper()
    if code not in BASE_RING:
        raise ValueError(f"Unsupported RNA base for suite frame: {base!r}")
    required = set(BASE_RING[code]) | {"C1'"}
    missing = required - atom_coordinates.keys()
    if missing:
        raise ValueError(f"RNA {code} residue lacks base-frame atoms: {sorted(missing)}")
    ring = np.stack([atom_coordinates[name] for name in BASE_RING[code]])
    glycosidic_atom, ring_carbon = FRAME_ANCHORS[code]
    n_atom = atom_coordinates[glycosidic_atom]
    c1_atom = atom_coordinates["C1'"]
    c_atom = atom_coordinates[ring_carbon]
    z_axis = np.cross(n_atom - c1_atom, n_atom - c_atom)
    z_norm = np.linalg.norm(z_axis)
    if z_norm <= np.finfo(float).eps:
        raise ValueError(f"RNA {code} frame anchor atoms are collinear")
    z_axis /= z_norm
    glycosidic_vector = c1_atom - n_atom
    glycosidic_norm = np.linalg.norm(glycosidic_vector)
    if glycosidic_norm <= np.finfo(float).eps:
        raise ValueError(f"RNA {code} glycosidic bond vector is degenerate")
    glycosidic_vector /= glycosidic_norm

    def rotate(vector: np.ndarray, angle: float) -> np.ndarray:
        return (
            vector * np.cos(angle)
            + np.cross(z_axis, vector) * np.sin(angle)
            + z_axis * np.dot(z_axis, vector) * (1 - np.cos(angle))
        )

    origin = n_atom + rotate(glycosidic_vector, np.deg2rad(141.478)) * 4.702
    x_axis = rotate(glycosidic_vector, np.deg2rad(-54.418))
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    frame = np.column_stack((x_axis, y_axis, z_axis))
    if not np.isfinite(frame).all() or np.linalg.det(frame) < 1.0 - 1e-10:
        raise ValueError(f"RNA {code} base frame is not a finite right-handed frame")
    return origin, frame


def _as_string(value: str) -> str:
    import gemmi

    return gemmi.cif.as_string(value).strip()


def _atom_coordinates(residue: Any) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    coordinates: dict[str, np.ndarray] = {}
    elements: dict[str, str] = {}
    occupancies: dict[str, float] = {}
    for atom in residue:
        name = atom.name.strip()
        if name in coordinates and atom.occ < occupancies[name]:
            continue
        point = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)
        if not np.isfinite(point).all():
            raise ValueError(f"{residue.name}:{residue.seqid} atom {name} has non-finite coordinates")
        coordinates[name] = point
        elements[name] = atom.element.name.upper()
        occupancies[name] = atom.occ
    return coordinates, elements


def _component_heavy_graph(
    base: str,
    *,
    five_prime: str = "phosphodiester",
    three_prime: str = "phosphodiester",
) -> tuple[set[str], list[dict[str, str]], dict[str, str]]:
    from mdna.chemistry import derive_rna_component_state

    state = derive_rna_component_state(
        f"RNA:{base}", five_prime=five_prime, three_prime=three_prime
    )
    heavy_names = {atom.name for atom in state.atoms if atom.element.upper() != "H"}
    bonds = [
        {"atom_1": bond.atom_1, "atom_2": bond.atom_2, "order": bond.order}
        for bond in state.bonds
        if bond.atom_1 in heavy_names and bond.atom_2 in heavy_names
    ]
    provenance = {
        "component_id": state.component_id,
        "component_sha256": state.component_sha256,
        "state_id": state.state_id,
        "evidence_sha256": state.evidence_sha256,
    }
    return heavy_names, bonds, provenance


def extract_suite(
    entry_id: str,
    chain_id: str,
    upstream: Any,
    downstream: Any,
    source_hash: str,
    chain_sequence: str,
    *,
    terminal_role: str | None = None,
) -> dict[str, Any]:
    upstream_base = upstream.name.strip().upper()
    downstream_base = downstream.name.strip().upper()
    internal_state = ("phosphodiester", "phosphodiester")
    if terminal_role is None:
        upstream_state = downstream_state = internal_state
    elif terminal_role == "five_prime":
        upstream_state = ("hydroxyl", "phosphodiester")
        downstream_state = internal_state
    elif terminal_role == "three_prime":
        upstream_state = internal_state
        downstream_state = ("phosphodiester", "hydroxyl")
    else:
        raise ValueError("terminal_role must be 'five_prime', 'three_prime', or None.")

    upstream_coordinates, upstream_elements = _atom_coordinates(upstream)
    downstream_coordinates, downstream_elements = _atom_coordinates(downstream)
    upstream_expected, upstream_bonds, upstream_provenance = _component_heavy_graph(
        upstream_base, five_prime=upstream_state[0], three_prime=upstream_state[1]
    )
    downstream_expected, downstream_bonds, downstream_provenance = _component_heavy_graph(
        downstream_base, five_prime=downstream_state[0], three_prime=downstream_state[1]
    )
    for residue, coordinates, elements, expected in (
        (upstream, upstream_coordinates, upstream_elements, upstream_expected),
        (downstream, downstream_coordinates, downstream_elements, downstream_expected),
    ):
        observed_heavy = {name for name, element in elements.items() if element != "H"}
        missing = sorted(expected - observed_heavy)
        unexpected = sorted(observed_heavy - expected)
        if missing or unexpected:
            raise ValueError(
                f"{entry_id}:{chain_id}:{residue.seqid} heavy-atom mismatch; "
                f"missing={missing}, unexpected={unexpected}"
            )

    upstream_origin, upstream_frame = make_base_frame(upstream_base, upstream_coordinates)
    downstream_origin, downstream_frame = make_base_frame(downstream_base, downstream_coordinates)

    atoms = []
    for residue_index, coordinates, elements, expected in (
        (0, upstream_coordinates, upstream_elements, upstream_expected),
        (1, downstream_coordinates, downstream_elements, downstream_expected),
    ):
        for name in sorted(expected):
            local = (coordinates[name] - upstream_origin) @ upstream_frame
            atoms.append(
                {
                    "residue_index": residue_index,
                    "atom_name": name,
                    "element": elements[name],
                    "xyz_local_A": [float(value) for value in local],
                }
            )

    bonds = [
        {"residue_index_1": 0, **bond, "residue_index_2": 0}
        for bond in upstream_bonds
    ] + [
        {"residue_index_1": 1, **bond, "residue_index_2": 1}
        for bond in downstream_bonds
    ]
    bonds.append(
        {
            "residue_index_1": 0,
            "atom_1": "O3'",
            "residue_index_2": 1,
            "atom_2": "P",
            "order": "single",
        }
    )
    upstream_o3 = upstream_coordinates["O3'"]
    downstream_p = downstream_coordinates["P"]
    link_distance = float(np.linalg.norm(upstream_o3 - downstream_p))
    relative_rotation = upstream_frame.T @ downstream_frame
    relative_translation = (downstream_origin - upstream_origin) @ upstream_frame
    return {
        "suite_id": f"{entry_id.upper()}:{chain_id}:{upstream.seqid}-{downstream.seqid}",
        "sequence": upstream_base + downstream_base,
        "source_entry": entry_id.upper(),
        "source_coordinate_sha256": source_hash,
        "chain_id": chain_id,
        "source_chain_sequence": chain_sequence,
        "source_residues": [str(upstream.seqid), str(downstream.seqid)],
        "residue_states": [
            upstream_provenance["state_id"], downstream_provenance["state_id"]
        ],
        "component_state_sources": [upstream_provenance, downstream_provenance],
        "coordinate_unit": "angstrom",
        "coordinate_frame": "upstream Tsukuba base frame from mdna.geometry.ReferenceBase, in angstrom",
        "downstream_base_frame_relative_to_upstream": {
            "translation_A": [float(value) for value in relative_translation],
            "rotation_matrix": relative_rotation.tolist(),
        },
        "observed_O3_prime_to_next_P_A": link_distance,
        "atoms": atoms,
        "bonds": bonds,
        **({"terminal_role": terminal_role} if terminal_role is not None else {}),
    }


def validate_suite(suite: dict[str, Any]) -> None:
    """Reject incomplete or malformed suite fragments before serialization."""

    atom_keys = {
        (atom["residue_index"], atom["atom_name"])
        for atom in suite["atoms"]
    }
    if len(atom_keys) != len(suite["atoms"]):
        raise ValueError(f"{suite['suite_id']}: duplicate atom key")
    if not all(np.isfinite(atom["xyz_local_A"]).all() for atom in suite["atoms"]):
        raise ValueError(f"{suite['suite_id']}: non-finite atom coordinates")
    bond_keys = set()
    for bond in suite["bonds"]:
        key_1 = (bond["residue_index_1"], bond["atom_1"])
        key_2 = (bond["residue_index_2"], bond["atom_2"])
        if key_1 not in atom_keys or key_2 not in atom_keys:
            raise ValueError(f"{suite['suite_id']}: bond endpoint is absent")
        canonical = tuple(sorted((key_1, key_2)))
        if canonical in bond_keys:
            raise ValueError(f"{suite['suite_id']}: duplicate bond {canonical}")
        bond_keys.add(canonical)
    if (0, "O3'") not in atom_keys or (1, "P") not in atom_keys:
        raise ValueError(f"{suite['suite_id']}: phosphodiester ports are absent")
    role = suite.get("terminal_role")
    states = tuple(suite.get("residue_states", ()))
    internal_state = "5prime_phosphodiester+3prime_phosphodiester"
    expected_states = {
        None: (internal_state, internal_state),
        "five_prime": ("5prime_hydroxyl+3prime_phosphodiester", internal_state),
        "three_prime": (internal_state, "5prime_phosphodiester+3prime_hydroxyl"),
    }
    if role not in expected_states or states != expected_states[role]:
        raise ValueError(f"{suite['suite_id']}: terminal role and residue states disagree")
    if not np.isfinite(suite["observed_O3_prime_to_next_P_A"]):
        raise ValueError(f"{suite['suite_id']}: non-finite O3'-P distance")
    rotation = np.asarray(
        suite["downstream_base_frame_relative_to_upstream"]["rotation_matrix"],
        dtype=float,
    )
    if (
        rotation.shape != (3, 3)
        or not np.isfinite(rotation).all()
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
        or np.linalg.det(rotation) < 1.0 - 1e-7
    ):
        raise ValueError(f"{suite['suite_id']}: downstream base transform is not a proper rotation")


def summarize_conformer_spread(suites: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for suite in suites:
        grouped.setdefault(suite["sequence"], []).append(suite)
    summary = {}
    for sequence, records in sorted(grouped.items()):
        coordinate_maps = [
            {
                (atom["residue_index"], atom["atom_name"]): np.asarray(atom["xyz_local_A"])
                for atom in suite["atoms"]
            }
            for suite in records
        ]
        pairwise_rmsd = []
        for left_index, right_index in combinations(range(len(records)), 2):
            left = coordinate_maps[left_index]
            right = coordinate_maps[right_index]
            if left.keys() != right.keys():
                raise ValueError(f"{sequence}: conformer atom sets differ")
            keys = sorted(left)
            delta = np.stack([left[key] - right[key] for key in keys])
            pairwise_rmsd.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
        values = np.asarray(pairwise_rmsd, dtype=float)
        summary[sequence] = {
            "suite_count": len(records),
            "pairwise_conformer_rmsd_count": len(pairwise_rmsd),
            "pairwise_conformer_rmsd_A": {
                "min": float(values.min()) if values.size else None,
                "median": float(np.median(values)) if values.size else None,
                "p95": float(np.percentile(values, 95)) if values.size else None,
                "max": float(values.max()) if values.size else None,
            },
        }
    return summary


def build_library(reference_dir: Path) -> dict[str, Any]:
    import gemmi

    manifest = json.loads((reference_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    manifest_hashes = {record["file"]: record["sha256"] for record in manifest["files"]}
    suites = []
    sources = []
    for entry_id in ENTRY_IDS:
        path = reference_dir / f"{entry_id}.cif"
        actual_hash = sha256(path)
        if manifest_hashes.get(path.name) != actual_hash:
            raise ValueError(f"{path.name}: SHA-256 does not match the reference manifest")
        structure = gemmi.read_structure(str(path))
        if len(structure) != 1:
            raise ValueError(f"{path.name}: expected one model, found {len(structure)}")
        sources.append({"file": path.name, "sha256": actual_hash})
        for chain in structure[0]:
            residues = [residue for residue in chain if residue.name.strip().upper() in BASE_RING]
            if not residues:
                continue
            sequence = "".join(residue.name.strip().upper() for residue in residues)
            if len(residues) < 4:
                continue
            # Keep only suites whose two residues are both internal in the source chain.
            for index in range(1, len(residues) - 2):
                suites.append(
                    extract_suite(
                        entry_id,
                        chain.name,
                        residues[index],
                        residues[index + 1],
                        actual_hash,
                        sequence,
                    )
                )
    for suite in suites:
        validate_suite(suite)
    counts = Counter(suite["sequence"] for suite in suites)
    all_dinucleotides = {first + second for first in BASE_RING for second in BASE_RING}
    link_distances = np.asarray(
        [suite["observed_O3_prime_to_next_P_A"] for suite in suites], dtype=float
    )
    return {
        "schema_version": 1,
        "library_type": "experimental internal RNA dinucleotide suites",
        "coordinate_unit": "angstrom",
        "coordinate_policy": "observed heavy-atom coordinates; no hydrogen coordinates synthesized",
        "atom_state_policy": "both residues use internal phosphodiester states; terminal source suites excluded",
        "frame_definition": "base-ring-derived local frame; not a standard base-pair or X3DNA frame",
        "source_manifest": "artifacts/rna_pilot/reference/ion_series/manifest.json",
        "source_manifest_sha256": sha256(reference_dir / "manifest.json"),
        "chemistry_registry": "mdna/data/rna_components.json",
        "chemistry_registry_sha256": sha256(PROJECT_ROOT / "mdna/data/rna_components.json"),
        "sources": sources,
        "summary": {
            "suite_count": len(suites),
            "sequence_type_count": len(counts),
            "suite_counts_by_sequence": dict(sorted(counts.items())),
            "unobserved_sequence_dinucleotides": sorted(all_dinucleotides - counts.keys()),
            "observed_O3_prime_to_next_P_A": {
                "count": int(len(link_distances)),
                "mean": float(link_distances.mean()),
                "min": float(link_distances.min()),
                "max": float(link_distances.max()),
            },
            "pairwise_conformer_spread_by_sequence": summarize_conformer_spread(suites),
        },
        "suites": suites,
        "interpretation_limit": (
            "This initial library contains internal suites from six crystal structures of one 8-nt sequence. "
            "It is a provenance-preserving source of observed conformers, not a general RNA conformer library, "
            "a path builder, or evidence of gap-closure performance."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=project_root / "artifacts/rna_pilot/reference/ion_series",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "artifacts/rna_pilot/reference/rna_suite_library.json",
    )
    args = parser.parse_args()
    report = build_library(args.reference_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
