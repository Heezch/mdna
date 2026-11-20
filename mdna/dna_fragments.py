"""Place source-traceable experimental DNA dinucleotide suites.

The initial library is finite and comes from one nucleosome structure family.
It supports validated suite lookup and placement into an explicit base frame;
it does not infer frames from a centerline or build a continuous DNA chain.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace as dataclass_replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


_BASES = frozenset("ACGT")
_UNIT_TO_ANGSTROM = {"angstrom": 1.0, "nm": 10.0}
_FRAME_TOLERANCE = 1e-3
_BASE_TO_COMPONENT = {"A": "DNA:DA", "C": "DNA:DC", "G": "DNA:DG", "T": "DNA:DT"}
_HEAVY_ELEMENTS = frozenset({"C", "N", "O", "P", "S"})


class UnsupportedDNASuiteError(ValueError):
    """Raised when a valid DNA context has no experimental suite in the library."""


@dataclass(frozen=True)
class DNAFragmentPlacement:
    """Coordinates for one suite placed in an explicit upstream base frame."""

    suite_id: str
    sequence: str
    atoms: tuple[dict[str, Any], ...]
    bonds: tuple[dict[str, Any], ...]
    downstream_frame: np.ndarray
    translation_residual_A: float | None
    rotation_residual_degrees: float | None


@dataclass(frozen=True)
class DNAFragmentPathSelection:
    """Best bounded sequence of observed suites for caller-provided base frames."""

    sequence: str
    placements: tuple[DNAFragmentPlacement, ...]
    shared_residue_rmsd_A: tuple[float, ...]
    normalized_score: float
    coordinate_unit: str
    input_base_frames_sha256: str
    source_files: tuple[dict[str, str], ...]
    max_translation_residual_A: float
    max_rotation_residual_degrees: float
    max_shared_residue_rmsd_A: float

    @property
    def suite_ids(self) -> tuple[str, ...]:
        return tuple(placement.suite_id for placement in self.placements)

    def to_report(self) -> dict[str, Any]:
        """Return JSON-compatible provenance, thresholds, and selected-path diagnostics."""
        return {
            "schema_version": 1,
            "algorithm": "dna-suite-dp-hard-limits-v1",
            "sequence": self.sequence,
            "coordinate_unit": self.coordinate_unit,
            "input_base_frames_sha256": self.input_base_frames_sha256,
            "source_files": [dict(source) for source in self.source_files],
            "limits": {
                "max_translation_residual_A": self.max_translation_residual_A,
                "max_rotation_residual_degrees": self.max_rotation_residual_degrees,
                "max_shared_residue_rmsd_A": self.max_shared_residue_rmsd_A,
            },
            "suite_ids": list(self.suite_ids),
            "placements": [
                {
                    "suite_id": placement.suite_id,
                    "sequence": placement.sequence,
                    "translation_residual_A": placement.translation_residual_A,
                    "rotation_residual_degrees": placement.rotation_residual_degrees,
                }
                for placement in self.placements
            ],
            "shared_residue_rmsd_A": list(self.shared_residue_rmsd_A),
            "normalized_score": self.normalized_score,
        }


@dataclass(frozen=True)
class DNAFragmentAssembly:
    """Merged heavy-atom DNA segment assembled from compatible suite placements."""

    sequence: str
    atoms: tuple[dict[str, Any], ...]
    bonds: tuple[dict[str, Any], ...]
    source_suite_ids: tuple[str, ...]
    five_prime_boundary_state: str
    five_prime_port: tuple[int, str] | None
    three_prime_boundary_state: str
    three_prime_port: tuple[int, str] | None
    shared_residue_rmsd_A: tuple[float, ...]
    shared_residue_max_displacement_A: float
    max_bond_length_deviation_A: float

    def to_mdtraj(self):
        """Build a one-frame MDTraj topology with explicit DNA component names."""
        import mdtraj as md

        if not isinstance(self.sequence, str) or not self.sequence or set(self.sequence) - _BASES:
            raise ValueError("assembled DNA sequence must contain standard A/C/G/T only.")
        if len(self.source_suite_ids) != len(self.sequence) - 1:
            raise ValueError("assembled DNA must record one source suite per dinucleotide.")
        if (self.five_prime_boundary_state != "open_phosphodiester_continuation"
                or self.five_prime_port != (0, "P")):
            raise ValueError("DNA 5' boundary must retain the first residue P continuation port.")
        expected_three_port = (len(self.sequence) - 1, "O3'")
        if (self.three_prime_boundary_state != "open_phosphodiester_continuation"
                or self.three_prime_port != expected_three_port):
            raise ValueError("DNA 3' boundary must retain the last residue O3' continuation port.")

        contracts = _registered_component_contracts()
        atom_records: dict[tuple[int, str], dict[str, Any]] = {}
        coordinates_A: dict[tuple[int, str], np.ndarray] = {}
        elements: dict[tuple[int, str], Any] = {}
        for atom in self.atoms:
            residue_index = atom.get("residue_index")
            atom_name = atom.get("atom_name")
            element_symbol = str(atom.get("element", "")).upper()
            key = (residue_index, atom_name)
            if (not isinstance(residue_index, int) or not 0 <= residue_index < len(self.sequence)
                    or not isinstance(atom_name, str) or not atom_name.strip() or key in atom_records):
                raise ValueError(f"assembled DNA has an invalid or duplicate atom record {key!r}.")
            expected_element = contracts[self.sequence[residue_index]]["atoms"].get(atom_name)
            if expected_element is None or expected_element != element_symbol:
                raise ValueError(f"assembled DNA atom {key!r} disagrees with its registered component.")
            xyz = np.asarray(atom.get("xyz_A"), dtype=float)
            if xyz.shape != (3,) or not np.isfinite(xyz).all():
                raise ValueError(f"assembled DNA atom {key!r} must have three finite angstrom coordinates.")
            try:
                element = md.element.get_by_symbol(element_symbol)
            except (KeyError, ValueError):
                element = None
            if element is None:
                raise ValueError(f"assembled DNA atom {key!r} has unsupported element {element_symbol!r}.")
            atom_records[key] = atom
            coordinates_A[key] = xyz
            elements[key] = element

        expected_residues = set(range(len(self.sequence)))
        if {index for index, _ in atom_records} != expected_residues:
            raise ValueError("assembled DNA atoms must cover every sequence residue.")
        for residue_index, base in enumerate(self.sequence):
            inventory = {name: elements[(residue_index, name)].symbol.upper()
                         for index, name in atom_records if index == residue_index}
            if inventory != contracts[base]["atoms"]:
                raise ValueError(f"assembled DNA residue {residue_index} atom inventory disagrees with its component.")

        expected_bonds = set()
        for residue_index, base in enumerate(self.sequence):
            expected_bonds.update(
                (tuple(sorted(((residue_index, first), (residue_index, second)))), order)
                for first, second, order in contracts[base]["bonds"]
            )
        expected_bonds.update(
            (tuple(sorted(((index, "O3'"), (index + 1, "P")))), "single")
            for index in range(len(self.sequence) - 1)
        )
        observed_bonds = set()
        for bond in self.bonds:
            first = (bond.get("residue_index_1"), bond.get("atom_1"))
            second = (bond.get("residue_index_2"), bond.get("atom_2"))
            if first not in atom_records or second not in atom_records or first == second:
                raise ValueError("assembled DNA bond has an invalid endpoint.")
            if bond.get("order") not in {"single", "double", "aromatic", "unspecified"}:
                raise ValueError("assembled DNA bond has an unsupported order label.")
            canonical = tuple(sorted((first, second)))
            if (canonical, bond["order"]) in observed_bonds:
                raise ValueError(f"assembled DNA contains duplicate bond {canonical!r}.")
            observed_bonds.add((canonical, bond["order"]))
        if observed_bonds != expected_bonds:
            raise ValueError("assembled DNA bond graph disagrees with component and phosphodiester contracts.")

        topology = md.Topology()
        chain = topology.add_chain(chain_id="A")
        atom_index_by_key = {}
        for residue_index, base in enumerate(self.sequence):
            residue = topology.add_residue(_BASE_TO_COMPONENT[base].split(":", 1)[1], chain,
                                           resSeq=residue_index + 1)
            for key in sorted(key for key in atom_records if key[0] == residue_index):
                atom_index_by_key[key] = topology.add_atom(key[1], elements[key], residue).index
        for bond in self.bonds:
            first = (bond["residue_index_1"], bond["atom_1"])
            second = (bond["residue_index_2"], bond["atom_2"])
            topology.add_bond(topology.atom(atom_index_by_key[first]),
                              topology.atom(atom_index_by_key[second]))
        xyz_nm = np.stack([coordinates_A[key] for key in atom_index_by_key], axis=0) / 10.0
        return md.Trajectory(xyz_nm[np.newaxis, :, :], topology)

    def save_mmcif(self, filename: str | Path) -> Path:
        """Write the fragment's atom records and explicit covalent graph."""
        from .structure_export import save_fragment_mmcif

        return save_fragment_mmcif(self, filename, polymer_type="polydeoxyribonucleotide")

    def save_pdb(self, filename: str | Path) -> Path:
        """Write DNA coordinates and explicit CONECT records for backbone links."""
        path = Path(filename)
        trajectory = self.to_mdtraj()
        if trajectory.n_atoms > 99999:
            raise ValueError("PDB atom serial numbers cannot represent more than 99999 atoms.")
        trajectory.save_pdb(str(path))
        serial_by_key = {
            (atom.residue.index, atom.name): atom.index + 1
            for atom in trajectory.topology.atoms
        }
        neighbors: dict[int, set[int]] = {}
        for bond in self.bonds:
            first = serial_by_key[(bond["residue_index_1"], bond["atom_1"])]
            second = serial_by_key[(bond["residue_index_2"], bond["atom_2"])]
            neighbors.setdefault(first, set()).add(second)
            neighbors.setdefault(second, set()).add(first)
        conect_lines = []
        for source, targets in sorted(neighbors.items()):
            ordered_targets = sorted(targets)
            for start in range(0, len(ordered_targets), 4):
                conect_lines.append(
                    f"CONECT{source:5d}" + "".join(
                        f"{target:5d}" for target in ordered_targets[start:start + 4]
                    )
                )
        lines = path.read_text(encoding="ascii").splitlines()
        end_index = next((index for index in range(len(lines) - 1, -1, -1)
                          if lines[index].startswith("END")), len(lines))
        lines[end_index:end_index] = conect_lines
        path.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
        return path


@dataclass(frozen=True)
class DNAGapClosureCandidate:
    """One source-suite path placed between two fixed DNA anchor residues."""

    path_selection: DNAFragmentPathSelection
    assembly: DNAFragmentAssembly
    endpoint_translation_residual_A: float
    endpoint_rotation_residual_degrees: float
    start_anchor_max_displacement_A: float
    end_anchor_max_displacement_A: float
    start_connection_reference_length_A: float
    end_connection_reference_length_A: float
    start_connection_length_A: float
    end_connection_length_A: float
    score_components: Mapping[str, float]
    normalized_score: float

    @property
    def suite_ids(self) -> tuple[str, ...]:
        return self.path_selection.suite_ids

    def to_report(self) -> dict[str, Any]:
        return {
            "suite_ids": list(self.suite_ids),
            "endpoint_translation_residual_A": self.endpoint_translation_residual_A,
            "endpoint_rotation_residual_degrees": self.endpoint_rotation_residual_degrees,
            "start_anchor_max_displacement_A": self.start_anchor_max_displacement_A,
            "end_anchor_max_displacement_A": self.end_anchor_max_displacement_A,
            "start_connection_reference_length_A": self.start_connection_reference_length_A,
            "end_connection_reference_length_A": self.end_connection_reference_length_A,
            "start_connection_length_A": self.start_connection_length_A,
            "end_connection_length_A": self.end_connection_length_A,
            "score_components": dict(self.score_components),
            "normalized_score": self.normalized_score,
            "assembly_atom_count": len(self.assembly.atoms),
            "assembly_bond_count": len(self.assembly.bonds),
            "assembly_max_bond_length_deviation_A": self.assembly.max_bond_length_deviation_A,
            "environment_screening": "not_evaluated",
        }


@dataclass(frozen=True)
class DNAGapClosureResult:
    """Bounded one-residue DNA gap search outcome; not scientific acceptance."""

    sequence: str
    candidates: tuple[DNAGapClosureCandidate, ...]
    status: str
    search_complete: bool
    expanded_states: int
    total_compatible_closures: int
    candidates_truncated: bool
    start_frame_sha256: str
    end_frame_sha256: str
    start_anchor_atoms_sha256: str
    end_anchor_atoms_sha256: str
    source_files: tuple[dict[str, str], ...]
    limits: Mapping[str, float | int]
    rejections: Mapping[str, int]
    trial_metric_minima: Mapping[str, float]
    trial_metric_evaluated_counts: Mapping[str, int]
    assembly_rejection_reasons: Mapping[str, int]

    @property
    def missing_sequence(self) -> str:
        return self.sequence[1]

    def to_report(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "algorithm": "dna-two-suite-anchor-enumeration-v1",
            "sequence_including_anchors": self.sequence,
            "missing_sequence": self.missing_sequence,
            "missing_residue_indices": [1],
            "status": self.status,
            "search_complete": self.search_complete,
            "expanded_states": self.expanded_states,
            "total_compatible_closures": self.total_compatible_closures,
            "returned_candidate_count": len(self.candidates),
            "candidates_truncated": self.candidates_truncated,
            "start_frame_sha256": self.start_frame_sha256,
            "end_frame_sha256": self.end_frame_sha256,
            "start_anchor_atoms_sha256": self.start_anchor_atoms_sha256,
            "end_anchor_atoms_sha256": self.end_anchor_atoms_sha256,
            "source_files": [dict(source) for source in self.source_files],
            "limits": dict(self.limits),
            "rejections": dict(self.rejections),
            "trial_metric_minima": dict(self.trial_metric_minima),
            "trial_metric_evaluated_counts": dict(self.trial_metric_evaluated_counts),
            "assembly_rejection_reasons": dict(self.assembly_rejection_reasons),
            "candidates": [candidate.to_report() for candidate in self.candidates],
            "environment_screening": "not_evaluated",
            "scientific_acceptance": False,
        }


class NoCompatibleDNASuitePathError(ValueError):
    """Raised when observed DNA suites cannot satisfy explicit path limits."""


@lru_cache(maxsize=1)
def _registered_component_contracts() -> dict[str, dict[str, Any]]:
    """Load registered DNA heavy-atom inventories and bonds once per process."""
    from .chemistry import get_nucleotide_component

    contracts = {}
    for base in sorted(_BASES):
        component = get_nucleotide_component(base)
        atoms = {atom.name: atom.element.upper() for atom in component.atoms
                 if atom.element.upper() != "H"}
        bonds = {
            tuple(sorted((bond.atom_1, bond.atom_2))) + (bond.order,)
            for bond in component.bonds
            if bond.atom_1 in atoms and bond.atom_2 in atoms
        }
        contracts[base] = {"sha256": component.sha256, "atoms": atoms, "bonds": bonds}
    return contracts


def _validated_frame(frame: Any, *, scale_to_angstrom: float, name: str) -> np.ndarray:
    array = np.asarray(frame, dtype=float)
    if array.shape != (4, 3):
        raise ValueError(f"{name} must have shape (4, 3): origin followed by three basis vectors.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite coordinates.")
    basis = array[1:]
    if not np.allclose(basis @ basis.T, np.eye(3), atol=_FRAME_TOLERANCE, rtol=0.0):
        raise ValueError(f"{name} basis must be orthonormal.")
    if np.linalg.det(basis) <= 0:
        raise ValueError(f"{name} basis must be right-handed.")
    result = array.copy()
    result[0] *= scale_to_angstrom
    return result


def _frame_digest(frame: np.ndarray, coordinate_unit: str) -> str:
    digest = hashlib.sha256()
    digest.update(coordinate_unit.encode("ascii"))
    digest.update(np.asarray(frame.shape, dtype="<i8").tobytes())
    digest.update(np.asarray(frame, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def _validate_dna_anchor_atoms(anchor_atoms: Mapping[str, Any], base: str, *, name: str
                              ) -> dict[str, np.ndarray]:
    if not isinstance(anchor_atoms, Mapping):
        raise TypeError(f"{name} must map DNA atom names to angstrom coordinates.")
    expected = set(_registered_component_contracts()[base]["atoms"])
    if set(anchor_atoms) != expected:
        raise ValueError(
            f"{name} atom names must exactly match the registered heavy-atom inventory; "
            f"missing={sorted(expected-set(anchor_atoms))}, unexpected={sorted(set(anchor_atoms)-expected)}."
        )
    coordinates = {}
    for atom_name, value in anchor_atoms.items():
        xyz = np.asarray(value, dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError(f"{name}[{atom_name!r}] must contain three finite angstrom coordinates.")
        coordinates[atom_name] = xyz
    return coordinates


def _dna_anchor_digest(anchor_atoms: Mapping[str, np.ndarray]) -> str:
    payload = {name: [float(value) for value in xyz]
               for name, xyz in sorted(anchor_atoms.items())}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"))
                          .encode("utf-8")).hexdigest()


def _placement_anchor_displacement(placement: DNAFragmentPlacement, residue_index: int,
                                   anchor_atoms: Mapping[str, np.ndarray]) -> float | None:
    placed = {atom["atom_name"]: np.asarray(atom["xyz_A"], dtype=float)
              for atom in placement.atoms if atom["residue_index"] == residue_index}
    if set(placed) != set(anchor_atoms):
        return None
    return max((float(np.linalg.norm(placed[name] - anchor_atoms[name]))
                for name in anchor_atoms), default=0.0)


def _validate_suite(record: Mapping[str, Any], source_hashes: set[str]) -> dict[str, Any]:
    suite = copy.deepcopy(dict(record))
    suite_id = suite.get("suite_id")
    sequence = suite.get("sequence")
    if not isinstance(suite_id, str) or not suite_id.strip():
        raise ValueError("Each DNA suite must have a non-empty suite_id.")
    if not isinstance(sequence, str) or len(sequence) != 2 or set(sequence) - _BASES:
        raise ValueError(f"DNA suite {suite_id!r} sequence must contain two standard A/C/G/T bases.")
    if suite.get("coordinate_unit") != "angstrom":
        raise ValueError(f"DNA suite {suite_id!r} coordinates must use angstrom units.")
    source_hash = suite.get("source_coordinate_sha256")
    if not isinstance(source_hash, str) or source_hash not in source_hashes:
        raise ValueError(f"DNA suite {suite_id!r} has no hash-bound coordinate source.")
    expected_components = [_BASE_TO_COMPONENT[base] for base in sequence]
    if suite.get("component_ids") != expected_components:
        raise ValueError(f"DNA suite {suite_id!r} component IDs do not match its sequence.")
    component_hashes = suite.get("component_sha256")
    if (not isinstance(component_hashes, list) or len(component_hashes) != 2
            or any(not isinstance(value, str) or len(value) != 64 for value in component_hashes)):
        raise ValueError(f"DNA suite {suite_id!r} requires two component template SHA-256 values.")
    component_contracts = _registered_component_contracts()

    atoms = suite.get("atoms")
    if not isinstance(atoms, list) or not atoms:
        raise ValueError(f"DNA suite {suite_id!r} must contain an atom list.")
    atom_keys: set[tuple[int, str]] = set()
    copied_atoms = []
    for atom in atoms:
        if not isinstance(atom, Mapping):
            raise ValueError(f"DNA suite {suite_id!r} has a malformed atom record.")
        residue_index, atom_name = atom.get("residue_index"), atom.get("atom_name")
        if residue_index not in (0, 1) or not isinstance(atom_name, str) or not atom_name.strip():
            raise ValueError(f"DNA suite {suite_id!r} atom must identify residue 0/1 and a name.")
        key = (residue_index, atom_name)
        if key in atom_keys:
            raise ValueError(f"DNA suite {suite_id!r} contains duplicate atom {key!r}.")
        element = str(atom.get("element", "")).upper()
        if element not in _HEAVY_ELEMENTS:
            raise ValueError(f"DNA suite {suite_id!r} atom {key!r} has unsupported element {element!r}.")
        xyz = np.asarray(atom.get("xyz_local_A"), dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError(f"DNA suite {suite_id!r} atom {key!r} needs three finite local coordinates.")
        copied = dict(atom)
        copied["element"] = element
        copied["xyz_local_A"] = xyz.tolist()
        atom_keys.add(key)
        copied_atoms.append(copied)
    if {index for index, _ in atom_keys} != {0, 1}:
        raise ValueError(f"DNA suite {suite_id!r} must contain atoms from both residues.")
    if (0, "O3'") not in atom_keys or (1, "P") not in atom_keys:
        raise ValueError(f"DNA suite {suite_id!r} must include upstream O3' and downstream P ports.")

    observed_atom_contracts = {
        index: {atom["atom_name"]: atom["element"] for atom in copied_atoms
               if atom["residue_index"] == index}
        for index in (0, 1)
    }
    observed_bond_contracts = {index: set() for index in (0, 1)}
    for bond in suite.get("bonds", []):
        if not isinstance(bond, Mapping):
            continue
        first = (bond.get("residue_index_1"), bond.get("atom_1"))
        second = (bond.get("residue_index_2"), bond.get("atom_2"))
        if first[0] == second[0] and first[0] in (0, 1):
            observed_bond_contracts[first[0]].add(
                tuple(sorted((first[1], second[1]))) + (bond.get("order"),)
            )
    for index, base in enumerate(sequence):
        expected = component_contracts[base]
        if component_hashes[index] != expected["sha256"]:
            raise ValueError(f"DNA suite {suite_id!r} component template hash disagrees with the registry.")
        if observed_atom_contracts[index] != expected["atoms"]:
            raise ValueError(f"DNA suite {suite_id!r} residue {index} atom inventory disagrees with the registry.")
        if observed_bond_contracts[index] != expected["bonds"]:
            raise ValueError(f"DNA suite {suite_id!r} residue {index} bond inventory disagrees with the registry.")

    bonds = suite.get("bonds")
    if not isinstance(bonds, list) or not bonds:
        raise ValueError(f"DNA suite {suite_id!r} must contain a bond list.")
    copied_bonds = []
    bond_keys = set()
    inter_residue_links = 0
    for bond in bonds:
        if not isinstance(bond, Mapping):
            raise ValueError(f"DNA suite {suite_id!r} has a malformed bond record.")
        first = (bond.get("residue_index_1"), bond.get("atom_1"))
        second = (bond.get("residue_index_2"), bond.get("atom_2"))
        if first not in atom_keys or second not in atom_keys or first == second:
            raise ValueError(f"DNA suite {suite_id!r} bond has an invalid endpoint.")
        canonical = tuple(sorted((first, second)))
        if canonical in bond_keys:
            raise ValueError(f"DNA suite {suite_id!r} contains duplicate bond {canonical!r}.")
        bond_keys.add(canonical)
        if first[0] != second[0]:
            if {first, second} != {(0, "O3'"), (1, "P")}:
                raise ValueError(f"DNA suite {suite_id!r} has an unsupported inter-residue bond.")
            inter_residue_links += 1
        if not isinstance(bond.get("order"), str) or not bond["order"]:
            raise ValueError(f"DNA suite {suite_id!r} bond order must be a non-empty string.")
        copied_bonds.append(dict(bond))
    if inter_residue_links != 1:
        raise ValueError(f"DNA suite {suite_id!r} must contain exactly one O3'-P inter-residue bond.")

    transform = suite.get("downstream_base_frame_relative_to_upstream")
    if not isinstance(transform, Mapping):
        raise ValueError(f"DNA suite {suite_id!r} lacks its downstream frame transform.")
    translation = np.asarray(transform.get("translation_A"), dtype=float)
    rotation = np.asarray(transform.get("rotation_matrix"), dtype=float)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError(f"DNA suite {suite_id!r} has an invalid frame translation.")
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=_FRAME_TOLERANCE, rtol=0.0)
            or np.linalg.det(rotation) <= 0):
        raise ValueError(f"DNA suite {suite_id!r} frame rotation must be right-handed and orthonormal.")
    link_distance = suite.get("observed_O3_prime_to_next_P_A")
    if not isinstance(link_distance, (int, float)) or not np.isfinite(link_distance) or link_distance <= 0:
        raise ValueError(f"DNA suite {suite_id!r} has an invalid observed O3'-P distance.")
    coordinate_by_key = {
        (atom["residue_index"], atom["atom_name"]): np.asarray(atom["xyz_local_A"], dtype=float)
        for atom in copied_atoms
    }
    measured_link_distance = float(np.linalg.norm(
        coordinate_by_key[(0, "O3'")] - coordinate_by_key[(1, "P")]
    ))
    if not np.isclose(measured_link_distance, link_distance, atol=1e-6, rtol=0.0):
        raise ValueError(
            f"DNA suite {suite_id!r} O3'-P distance does not match suite coordinates."
        )

    suite["atoms"] = copied_atoms
    suite["bonds"] = copied_bonds
    suite["downstream_base_frame_relative_to_upstream"] = {
        "translation_A": translation.tolist(), "rotation_matrix": rotation.tolist()
    }
    return suite


class DNAFragmentLibrary:
    """Validated experimental DNA suites with explicit finite sequence coverage."""

    def __init__(self, suites: list[Mapping[str, Any]], *, source_files: tuple[dict[str, str], ...]):
        source_hashes = {source["sha256"] for source in source_files}
        if not source_hashes:
            raise ValueError("DNA fragment library must declare at least one source file.")
        if not suites:
            raise ValueError("DNA fragment library must contain at least one suite.")
        validated = [_validate_suite(record, source_hashes) for record in suites]
        ids = [record["suite_id"] for record in validated]
        if len(ids) != len(set(ids)):
            raise ValueError("DNA fragment library contains duplicate suite_id values.")
        self._suites = tuple(validated)
        self.source_files = copy.deepcopy(source_files)

    @classmethod
    def from_json(cls, path: str | Path) -> "DNAFragmentLibrary":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, Mapping) or data.get("schema_version") != 1:
            raise ValueError("Unsupported DNA fragment library schema_version; expected 1.")
        if data.get("library_type") != "experimental internal DNA dinucleotide suites":
            raise ValueError("DNA fragment library has an unsupported library_type.")
        if data.get("coordinate_unit") != "angstrom":
            raise ValueError("DNA fragment library coordinate_unit must be angstrom.")
        if data.get("sources_represent_independent_families") is not False:
            raise ValueError("DNA source-family independence must be explicitly reported as false.")
        source_files = data.get("source_files")
        if not isinstance(source_files, list) or not source_files:
            raise ValueError("DNA fragment library must declare source_files.")
        for source in source_files:
            if (not isinstance(source, Mapping) or not isinstance(source.get("file"), str)
                    or not isinstance(source.get("sha256"), str) or len(source["sha256"]) != 64):
                raise ValueError("DNA fragment library source_files entries require a file and SHA-256.")
        suites = data.get("suites")
        if not isinstance(suites, list):
            raise ValueError("DNA fragment library suites must be a list.")
        return cls(suites, source_files=tuple(dict(source) for source in source_files))

    @property
    def available_sequences(self) -> tuple[str, ...]:
        return tuple(sorted({suite["sequence"] for suite in self._suites}))

    def get_suites(self, sequence: str) -> tuple[dict[str, Any], ...]:
        if not isinstance(sequence, str) or len(sequence) != 2 or set(sequence) - _BASES:
            raise ValueError("sequence must contain two standard DNA A/C/G/T bases.")
        matches = tuple(copy.deepcopy(suite) for suite in self._suites if suite["sequence"] == sequence)
        if not matches:
            covered = ", ".join(self.available_sequences)
            raise UnsupportedDNASuiteError(
                f"No observed DNA suite for {sequence}; current library coverage: {covered}."
            )
        return matches

    def place_suite(self, suite: Mapping[str, Any], upstream_frame: Any, *,
                    coordinate_unit: str, downstream_frame: Any | None = None) -> DNAFragmentPlacement:
        """Place a registered suite at an explicit MDNA row-basis frame."""
        if coordinate_unit not in _UNIT_TO_ANGSTROM:
            raise ValueError("coordinate_unit must be 'angstrom' or 'nm'.")
        validated = _validate_suite(suite, {source["sha256"] for source in self.source_files})
        source = next((item for item in self._suites if item["suite_id"] == validated["suite_id"]), None)
        if source is None or source != validated:
            raise ValueError("suite must be an unmodified record obtained from this library.")
        upstream = _validated_frame(upstream_frame,
                                    scale_to_angstrom=_UNIT_TO_ANGSTROM[coordinate_unit],
                                    name="upstream_frame")
        target = None if downstream_frame is None else _validated_frame(
            downstream_frame, scale_to_angstrom=1.0, name="downstream_frame")

        origin = upstream[0]
        basis_rows = upstream[1:]
        atoms = []
        for atom in validated["atoms"]:
            xyz_A = np.asarray(atom["xyz_local_A"], dtype=float) @ basis_rows + origin
            placed = {key: value for key, value in atom.items() if key != "xyz_local_A"}
            placed["xyz_A"] = xyz_A.tolist()
            atoms.append(placed)

        relative = validated["downstream_base_frame_relative_to_upstream"]
        rotation = np.asarray(relative["rotation_matrix"], dtype=float)
        predicted_basis_rows = rotation.T @ basis_rows
        predicted_origin = origin + np.asarray(relative["translation_A"], dtype=float) @ basis_rows
        predicted_frame = np.vstack((predicted_origin, predicted_basis_rows))
        translation_residual = rotation_residual = None
        if target is not None:
            translation_residual = float(np.linalg.norm(predicted_origin - target[0]))
            relative_orientation = predicted_basis_rows @ target[1:].T
            cosine = np.clip((np.trace(relative_orientation) - 1) / 2, -1.0, 1.0)
            rotation_residual = float(np.degrees(np.arccos(cosine)))
        return DNAFragmentPlacement(
            suite_id=validated["suite_id"], sequence=validated["sequence"],
            atoms=tuple(atoms), bonds=tuple(dict(bond) for bond in validated["bonds"]),
            downstream_frame=predicted_frame,
            translation_residual_A=translation_residual,
            rotation_residual_degrees=rotation_residual,
        )

    @staticmethod
    def _shared_residue_rmsd_A(first: DNAFragmentPlacement,
                               second: DNAFragmentPlacement) -> float | None:
        first_atoms = {
            atom["atom_name"]: np.asarray(atom["xyz_A"], dtype=float)
            for atom in first.atoms
            if atom["residue_index"] == 1 and atom.get("element", "").upper() != "H"
        }
        second_atoms = {
            atom["atom_name"]: np.asarray(atom["xyz_A"], dtype=float)
            for atom in second.atoms
            if atom["residue_index"] == 0 and atom.get("element", "").upper() != "H"
        }
        names = sorted(first_atoms.keys() & second_atoms.keys())
        if len(names) < 3:
            return None
        delta = np.stack([first_atoms[name] - second_atoms[name] for name in names])
        return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))

    def select_suite_path(self, sequence: str, base_frames: Any, *,
                          coordinate_unit: str,
                          max_translation_residual_A: float,
                          max_rotation_residual_degrees: float,
                          max_shared_residue_rmsd_A: float) -> DNAFragmentPathSelection:
        """Select observed suites under explicit frame and continuity limits.

        The dynamic program minimizes normalized frame residuals plus shared
        residue continuity. It does not deform suites, synthesize missing
        sequence contexts, or guarantee physical suitability outside the
        source family.
        """
        if not isinstance(sequence, str) or len(sequence) < 2 or set(sequence) - _BASES:
            raise ValueError("sequence must contain at least two standard DNA A/C/G/T bases.")
        if coordinate_unit not in _UNIT_TO_ANGSTROM:
            raise ValueError("coordinate_unit must be 'angstrom' or 'nm'.")
        limits = {
            "max_translation_residual_A": max_translation_residual_A,
            "max_rotation_residual_degrees": max_rotation_residual_degrees,
            "max_shared_residue_rmsd_A": max_shared_residue_rmsd_A,
        }
        for name, value in limits.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number.")
        raw_frames = np.asarray(base_frames, dtype=float)
        if raw_frames.shape != (len(sequence), 4, 3):
            raise ValueError(f"base_frames must have shape ({len(sequence)}, 4, 3).")
        frames_A = np.stack([
            _validated_frame(frame, scale_to_angstrom=_UNIT_TO_ANGSTROM[coordinate_unit],
                             name=f"base_frames[{index}]")
            for index, frame in enumerate(raw_frames)
        ])

        candidates_by_step: list[list[DNAFragmentPlacement]] = []
        for step in range(len(sequence) - 1):
            pair = sequence[step:step + 2]
            candidates = []
            for suite in self.get_suites(pair):
                placement = self.place_suite(
                    suite, frames_A[step], coordinate_unit="angstrom",
                    downstream_frame=frames_A[step + 1],
                )
                if (placement.translation_residual_A <= max_translation_residual_A
                        and placement.rotation_residual_degrees <= max_rotation_residual_degrees):
                    candidates.append(placement)
            if not candidates:
                raise NoCompatibleDNASuitePathError(
                    f"No DNA suite at step {step} ({pair}) satisfies the downstream frame limits."
                )
            candidates.sort(key=lambda placement: placement.suite_id)
            candidates_by_step.append(candidates)

        states = []
        for candidate in candidates_by_step[0]:
            translation = candidate.translation_residual_A / max_translation_residual_A
            rotation = candidate.rotation_residual_degrees / max_rotation_residual_degrees
            states.append({"cost": translation**2 + rotation**2,
                           "path": (candidate.suite_id,), "placement": candidate})
        for step in range(1, len(candidates_by_step)):
            next_states = []
            for candidate in candidates_by_step[step]:
                best = None
                translation = candidate.translation_residual_A / max_translation_residual_A
                rotation = candidate.rotation_residual_degrees / max_rotation_residual_degrees
                candidate_cost = translation**2 + rotation**2
                for previous in states:
                    shared = self._shared_residue_rmsd_A(previous["placement"], candidate)
                    if shared is None or shared > max_shared_residue_rmsd_A:
                        continue
                    option = {
                        "cost": previous["cost"] + candidate_cost +
                                (shared / max_shared_residue_rmsd_A) ** 2,
                        "path": previous["path"] + (candidate.suite_id,),
                        "placement": candidate,
                    }
                    if best is None or (option["cost"], option["path"]) < (best["cost"], best["path"]):
                        best = option
                if best is not None:
                    next_states.append(best)
            if not next_states:
                raise NoCompatibleDNASuitePathError(
                    f"No adjacent DNA suites at step {step} satisfy the shared-residue limit."
                )
            states = next_states

        best = min(states, key=lambda state: (state["cost"], state["path"]))
        placements = tuple(
            next(candidate for candidate in candidates_by_step[index]
                 if candidate.suite_id == suite_id)
            for index, suite_id in enumerate(best["path"])
        )
        shared = tuple(
            float(self._shared_residue_rmsd_A(placements[index], placements[index + 1]))
            for index in range(len(placements) - 1)
        )
        return DNAFragmentPathSelection(
            sequence=sequence, placements=placements,
            shared_residue_rmsd_A=shared, normalized_score=float(best["cost"]),
            coordinate_unit=coordinate_unit,
            input_base_frames_sha256=_frame_digest(raw_frames, coordinate_unit),
            source_files=tuple(copy.deepcopy(self.source_files)),
            max_translation_residual_A=float(max_translation_residual_A),
            max_rotation_residual_degrees=float(max_rotation_residual_degrees),
            max_shared_residue_rmsd_A=float(max_shared_residue_rmsd_A),
        )

    def assemble_suite_path(
        self,
        path: DNAFragmentPathSelection,
        *,
        max_shared_atom_displacement_A: float,
        max_bond_length_deviation_A: float,
    ) -> DNAFragmentAssembly:
        """Merge a selected suite path while enforcing caller-supplied limits.

        Shared coordinates are checked and the first copy is retained. This
        operation does not average coordinates, refine torsions, or establish
        that the resulting segment is an experimentally valid DNA model.
        """
        if not isinstance(path, DNAFragmentPathSelection):
            raise TypeError("path must be a DNAFragmentPathSelection.")
        if len(path.sequence) < 2 or len(path.placements) != len(path.sequence) - 1:
            raise ValueError("path must contain one suite placement per sequence dinucleotide.")
        if path.source_files != tuple(self.source_files):
            raise ValueError("path source files do not match this DNA fragment library.")
        limits = {
            "max_shared_atom_displacement_A": max_shared_atom_displacement_A,
            "max_bond_length_deviation_A": max_bond_length_deviation_A,
        }
        for name, value in limits.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number.")

        suite_by_id = {suite["suite_id"]: suite for suite in self._suites}
        for step, placement in enumerate(path.placements):
            source = suite_by_id.get(placement.suite_id)
            if source is None or source["sequence"] != path.sequence[step:step + 2]:
                raise ValueError(f"path suite at step {step} is not registered for its sequence.")
            expected_atoms = {
                (atom["residue_index"], atom["atom_name"], atom["element"])
                for atom in source["atoms"]
            }
            actual_atoms = {
                (atom.get("residue_index"), atom.get("atom_name"),
                 str(atom.get("element", "")).upper())
                for atom in placement.atoms
            }
            if len(actual_atoms) != len(placement.atoms) or actual_atoms != expected_atoms:
                raise ValueError(f"path suite at step {step} atom inventory differs from its library record.")
            expected_bonds = {
                (bond["residue_index_1"], bond["atom_1"], bond["residue_index_2"],
                 bond["atom_2"], bond["order"])
                for bond in source["bonds"]
            }
            actual_bonds = {
                (bond.get("residue_index_1"), bond.get("atom_1"),
                 bond.get("residue_index_2"), bond.get("atom_2"), bond.get("order"))
                for bond in placement.bonds
            }
            if len(actual_bonds) != len(placement.bonds) or actual_bonds != expected_bonds:
                raise ValueError(f"path suite at step {step} bond graph differs from its library record.")

        atom_records: dict[tuple[int, str], dict[str, Any]] = {}
        residue_inventories: dict[int, dict[str, str]] = {}
        shared_differences: dict[int, list[float]] = {}
        bond_records: dict[tuple[tuple[int, str], tuple[int, str]], dict[str, Any]] = {}
        bond_reference_lengths: list[tuple[tuple[tuple[int, str], tuple[int, str]], float]] = []

        for step, placement in enumerate(path.placements):
            local_atoms: dict[tuple[int, str], tuple[dict[str, Any], np.ndarray]] = {}
            local_inventories: dict[int, dict[str, str]] = {0: {}, 1: {}}
            for atom in placement.atoms:
                local_index, name = atom.get("residue_index"), atom.get("atom_name")
                element = str(atom.get("element", "")).upper()
                xyz = np.asarray(atom.get("xyz_A"), dtype=float)
                if (local_index not in (0, 1) or not isinstance(name, str) or not name
                        or xyz.shape != (3,) or not np.isfinite(xyz).all()):
                    raise ValueError(f"suite {placement.suite_id!r} has a malformed placed atom.")
                local_key = (local_index, name)
                if local_key in local_atoms:
                    raise ValueError(f"suite {placement.suite_id!r} contains duplicate atom {local_key!r}.")
                local_atoms[local_key] = (atom, xyz)
                local_inventories[local_index][name] = element
                global_key = (step + local_index, name)
                copied = dict(atom)
                copied["residue_index"] = step + local_index
                copied["xyz_A"] = xyz.tolist()
                existing = atom_records.get(global_key)
                if existing is None:
                    atom_records[global_key] = copied
                else:
                    if str(existing.get("element", "")).upper() != element:
                        raise ValueError(f"shared atom {global_key!r} has conflicting elements.")
                    displacement = float(np.linalg.norm(np.asarray(existing["xyz_A"]) - xyz))
                    shared_differences.setdefault(step, []).append(displacement)
                    if displacement > max_shared_atom_displacement_A:
                        raise ValueError(
                            f"shared atom displacement at residue {step} is {displacement:.6g} Å, "
                            f"above limit {max_shared_atom_displacement_A:.6g} Å."
                        )

            for local_index, inventory in local_inventories.items():
                global_index = step + local_index
                previous = residue_inventories.get(global_index)
                if previous is not None and previous != inventory:
                    raise ValueError(f"shared residue {global_index} atom inventory or elements differ between suites.")
                residue_inventories[global_index] = inventory

            for bond in placement.bonds:
                first_local = (bond.get("residue_index_1"), bond.get("atom_1"))
                second_local = (bond.get("residue_index_2"), bond.get("atom_2"))
                if first_local not in local_atoms or second_local not in local_atoms:
                    raise ValueError(f"suite {placement.suite_id!r} bond endpoint is absent.")
                first_global = (step + first_local[0], first_local[1])
                second_global = (step + second_local[0], second_local[1])
                canonical = tuple(sorted((first_global, second_global)))
                order = bond.get("order")
                existing_bond = bond_records.get(canonical)
                if existing_bond is not None and existing_bond["order"] != order:
                    raise ValueError(f"bond {canonical!r} has conflicting bond orders.")
                if existing_bond is None:
                    bond_records[canonical] = {
                        "residue_index_1": canonical[0][0], "atom_1": canonical[0][1],
                        "residue_index_2": canonical[1][0], "atom_2": canonical[1][1],
                        "order": order,
                    }
                first_xyz, second_xyz = local_atoms[first_local][1], local_atoms[second_local][1]
                bond_reference_lengths.append((canonical, float(np.linalg.norm(first_xyz - second_xyz))))

        five_port, three_port = (0, "P"), (len(path.sequence) - 1, "O3'")
        if five_port not in atom_records or three_port not in atom_records:
            raise ValueError("assembled DNA segment lacks a 5' P or 3' O3' continuation port.")
        deviations = []
        for canonical, reference_length in bond_reference_lengths:
            actual_length = float(np.linalg.norm(
                np.asarray(atom_records[canonical[0]]["xyz_A"])
                - np.asarray(atom_records[canonical[1]]["xyz_A"])
            ))
            deviation = abs(actual_length - reference_length)
            deviations.append(deviation)
            if deviation > max_bond_length_deviation_A:
                raise ValueError(
                    f"assembled bond length changed by {deviation:.6g} Å, above limit "
                    f"{max_bond_length_deviation_A:.6g} Å."
                )

        interfaces = tuple(
            float(np.sqrt(np.mean(np.square(shared_differences[index]))))
            for index in range(1, len(path.placements))
        )
        return DNAFragmentAssembly(
            sequence=path.sequence,
            atoms=tuple(atom_records[key] for key in sorted(atom_records)),
            bonds=tuple(bond_records[key] for key in sorted(bond_records)),
            source_suite_ids=tuple(placement.suite_id for placement in path.placements),
            five_prime_boundary_state="open_phosphodiester_continuation",
            five_prime_port=five_port,
            three_prime_boundary_state="open_phosphodiester_continuation",
            three_prime_port=three_port,
            shared_residue_rmsd_A=interfaces,
            shared_residue_max_displacement_A=max(
                (value for values in shared_differences.values() for value in values), default=0.0
            ),
            max_bond_length_deviation_A=max(deviations, default=0.0),
        )

    def close_gap_from_anchors(
        self,
        sequence: str,
        start_frame: Any,
        end_frame: Any,
        start_anchor_atoms: Mapping[str, Any],
        end_anchor_atoms: Mapping[str, Any],
        *,
        coordinate_unit: str,
        max_translation_residual_A: float,
        max_rotation_residual_degrees: float,
        max_shared_residue_rmsd_A: float,
        max_shared_atom_displacement_A: float,
        max_bond_length_deviation_A: float,
        max_anchor_atom_displacement_A: float,
        max_anchor_connection_length_deviation_A: float,
        max_search_states: int,
        max_returned_candidates: int,
    ) -> DNAGapClosureResult:
        """Enumerate bounded two-suite closures for one internal DNA residue.

        ``sequence`` contains the two fixed anchors and exactly one missing
        residue. Anchor maps must contain the complete registered heavy-atom
        inventory. Accepted output restores anchor coordinates exactly and
        rechecks every source bond against the assembled coordinates. This
        prototype does not evaluate environment clashes or base-pair geometry.
        """
        if not isinstance(sequence, str) or len(sequence) != 3 or set(sequence) - _BASES:
            raise ValueError("sequence must contain two anchors and exactly one standard DNA A/C/G/T gap base.")
        if coordinate_unit not in _UNIT_TO_ANGSTROM:
            raise ValueError("coordinate_unit must be 'angstrom' or 'nm'.")
        positive_limits = {
            "max_translation_residual_A": max_translation_residual_A,
            "max_rotation_residual_degrees": max_rotation_residual_degrees,
            "max_shared_residue_rmsd_A": max_shared_residue_rmsd_A,
        }
        nonnegative_limits = {
            "max_shared_atom_displacement_A": max_shared_atom_displacement_A,
            "max_bond_length_deviation_A": max_bond_length_deviation_A,
            "max_anchor_atom_displacement_A": max_anchor_atom_displacement_A,
            "max_anchor_connection_length_deviation_A": max_anchor_connection_length_deviation_A,
        }
        for name, value in {**positive_limits, **nonnegative_limits}.items():
            if not np.isfinite(value) or value < 0 or (name in positive_limits and value == 0):
                qualifier = "positive" if name in positive_limits else "non-negative"
                raise ValueError(f"{name} must be finite and {qualifier}.")
        for name, value in (("max_search_states", max_search_states),
                            ("max_returned_candidates", max_returned_candidates)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")

        start_A = _validated_frame(start_frame, scale_to_angstrom=_UNIT_TO_ANGSTROM[coordinate_unit],
                                   name="start_frame")
        end_A = _validated_frame(end_frame, scale_to_angstrom=_UNIT_TO_ANGSTROM[coordinate_unit],
                                 name="end_frame")
        start_atoms = _validate_dna_anchor_atoms(start_anchor_atoms, sequence[0],
                                                 name="start_anchor_atoms")
        end_atoms = _validate_dna_anchor_atoms(end_anchor_atoms, sequence[2],
                                               name="end_anchor_atoms")
        left_suites = tuple(sorted(self.get_suites(sequence[:2]), key=lambda item: item["suite_id"]))
        right_suites = tuple(sorted(self.get_suites(sequence[1:]), key=lambda item: item["suite_id"]))
        suite_by_id = {suite["suite_id"]: suite for suite in self._suites}
        rejections = {
            "start_anchor_inventory": 0,
            "start_anchor_displacement": 0,
            "anchor_displacement": 0,
            "endpoint_frame": 0,
            "shared_residue": 0,
            "end_anchor_inventory": 0,
            "end_anchor_displacement": 0,
            "anchor_connection_geometry": 0,
            "assembly": 0,
            "pinned_anchor_bond_geometry": 0,
        }
        trial_metric_minima: dict[str, float] = {}
        trial_metric_evaluated_counts: dict[str, int] = {}
        assembly_rejection_reasons: dict[str, int] = {}

        def observe_metric(name: str, value: float) -> None:
            value = float(value)
            trial_metric_minima[name] = min(value, trial_metric_minima.get(name, value))
            trial_metric_evaluated_counts[name] = trial_metric_evaluated_counts.get(name, 0) + 1

        expanded = 0
        total = 0
        complete = True
        candidates: list[DNAGapClosureCandidate] = []
        for left_suite in left_suites:
            for right_suite in right_suites:
                if expanded >= max_search_states:
                    complete = False
                    break
                expanded += 1
                left = self.place_suite(left_suite, start_A, coordinate_unit="angstrom")
                start_displacement = _placement_anchor_displacement(left, 0, start_atoms)
                if start_displacement is None:
                    rejections["start_anchor_inventory"] += 1
                    continue
                observe_metric("start_anchor_max_displacement_A", start_displacement)
                if start_displacement > max_anchor_atom_displacement_A:
                    rejections["start_anchor_displacement"] += 1
                    rejections["anchor_displacement"] += 1
                    continue

                right = self.place_suite(
                    right_suite, left.downstream_frame, coordinate_unit="angstrom",
                    downstream_frame=end_A,
                )
                assert right.translation_residual_A is not None
                assert right.rotation_residual_degrees is not None
                observe_metric("endpoint_translation_residual_A", right.translation_residual_A)
                observe_metric("endpoint_rotation_residual_degrees", right.rotation_residual_degrees)
                if (right.translation_residual_A > max_translation_residual_A
                        or right.rotation_residual_degrees > max_rotation_residual_degrees):
                    rejections["endpoint_frame"] += 1
                    continue
                shared_rmsd = self._shared_residue_rmsd_A(left, right)
                if shared_rmsd is not None:
                    observe_metric("shared_residue_rmsd_A", shared_rmsd)
                if shared_rmsd is None or shared_rmsd > max_shared_residue_rmsd_A:
                    rejections["shared_residue"] += 1
                    continue
                end_displacement = _placement_anchor_displacement(right, 1, end_atoms)
                if end_displacement is None:
                    rejections["end_anchor_inventory"] += 1
                    continue
                observe_metric("end_anchor_max_displacement_A", end_displacement)
                if end_displacement > max_anchor_atom_displacement_A:
                    rejections["end_anchor_displacement"] += 1
                    rejections["anchor_displacement"] += 1
                    continue

                left_phosphate = next(np.asarray(atom["xyz_A"], dtype=float)
                                      for atom in left.atoms
                                      if atom["residue_index"] == 1 and atom["atom_name"] == "P")
                right_o3 = next(np.asarray(atom["xyz_A"], dtype=float)
                                for atom in right.atoms
                                if atom["residue_index"] == 0 and atom["atom_name"] == "O3'")
                start_connection = float(np.linalg.norm(start_atoms["O3'"] - left_phosphate))
                end_connection = float(np.linalg.norm(right_o3 - end_atoms["P"]))
                left_reference = float(left_suite["observed_O3_prime_to_next_P_A"])
                right_reference = float(right_suite["observed_O3_prime_to_next_P_A"])
                start_connection_deviation = abs(start_connection - left_reference)
                end_connection_deviation = abs(end_connection - right_reference)
                observe_metric("max_anchor_connection_length_deviation_A",
                               max(start_connection_deviation, end_connection_deviation))
                if (abs(start_connection - left_reference) > max_anchor_connection_length_deviation_A
                        or abs(end_connection - right_reference) > max_anchor_connection_length_deviation_A):
                    rejections["anchor_connection_geometry"] += 1
                    continue

                frames = np.stack((start_A, left.downstream_frame, end_A))
                path = DNAFragmentPathSelection(
                    sequence=sequence,
                    placements=(left, right),
                    shared_residue_rmsd_A=(float(shared_rmsd),),
                    normalized_score=0.0,
                    coordinate_unit=coordinate_unit,
                    input_base_frames_sha256=_frame_digest(frames, coordinate_unit),
                    source_files=tuple(copy.deepcopy(self.source_files)),
                    max_translation_residual_A=float(max_translation_residual_A),
                    max_rotation_residual_degrees=float(max_rotation_residual_degrees),
                    max_shared_residue_rmsd_A=float(max_shared_residue_rmsd_A),
                )
                try:
                    assembly = self.assemble_suite_path(
                        path,
                        max_shared_atom_displacement_A=max_shared_atom_displacement_A,
                        max_bond_length_deviation_A=max_bond_length_deviation_A,
                    )
                except ValueError as exc:
                    rejections["assembly"] += 1
                    reason = str(exc)
                    assembly_rejection_reasons[reason] = assembly_rejection_reasons.get(reason, 0) + 1
                    continue

                pinned_atoms = []
                coordinates = {}
                for atom in assembly.atoms:
                    copied = dict(atom)
                    key = (atom["residue_index"], atom["atom_name"])
                    if atom["residue_index"] == 0:
                        copied["xyz_A"] = start_atoms[atom["atom_name"]].tolist()
                    elif atom["residue_index"] == 2:
                        copied["xyz_A"] = end_atoms[atom["atom_name"]].tolist()
                    coordinates[key] = np.asarray(copied["xyz_A"], dtype=float)
                    pinned_atoms.append(copied)

                source_bond_lengths = []
                for step, placement in enumerate((left, right)):
                    local_xyz = {(atom["residue_index"], atom["atom_name"]):
                                 np.asarray(atom["xyz_A"], dtype=float)
                                 for atom in placement.atoms}
                    for bond in placement.bonds:
                        first = (bond["residue_index_1"], bond["atom_1"])
                        second = (bond["residue_index_2"], bond["atom_2"])
                        global_first = (step + first[0], first[1])
                        global_second = (step + second[0], second[1])
                        reference = float(np.linalg.norm(local_xyz[first] - local_xyz[second]))
                        observed = float(np.linalg.norm(coordinates[global_first] - coordinates[global_second]))
                        source_bond_lengths.append(abs(observed - reference))
                pinned_bond_deviation = max(source_bond_lengths, default=0.0)
                observe_metric("pinned_anchor_max_bond_length_deviation_A", pinned_bond_deviation)
                if pinned_bond_deviation > max_bond_length_deviation_A:
                    rejections["pinned_anchor_bond_geometry"] += 1
                    continue
                assembly = dataclass_replace(
                    assembly,
                    atoms=tuple(pinned_atoms),
                    max_bond_length_deviation_A=float(pinned_bond_deviation),
                )

                score_components = {
                    "endpoint_translation": (right.translation_residual_A / max_translation_residual_A) ** 2,
                    "endpoint_rotation": (right.rotation_residual_degrees / max_rotation_residual_degrees) ** 2,
                    "shared_residue_continuity": (shared_rmsd / max_shared_residue_rmsd_A) ** 2,
                    "start_anchor_deviation": (start_displacement / max(1e-12, max_anchor_atom_displacement_A or 1.0)) ** 2,
                    "end_anchor_deviation": (end_displacement / max(1e-12, max_anchor_atom_displacement_A or 1.0)) ** 2,
                    "anchor_connection_deviation": (
                        (abs(start_connection - left_reference) / max(1e-12, max_anchor_connection_length_deviation_A or 1.0)) ** 2
                        + (abs(end_connection - right_reference) / max(1e-12, max_anchor_connection_length_deviation_A or 1.0)) ** 2
                    ),
                    "pinned_assembly_bond_deviation": (
                        pinned_bond_deviation / max(1e-12, max_bond_length_deviation_A or 1.0)
                    ) ** 2,
                }
                score = float(sum(score_components.values()))
                path = dataclass_replace(path, normalized_score=score)
                candidate = DNAGapClosureCandidate(
                    path_selection=path,
                    assembly=assembly,
                    endpoint_translation_residual_A=float(right.translation_residual_A),
                    endpoint_rotation_residual_degrees=float(right.rotation_residual_degrees),
                    start_anchor_max_displacement_A=float(start_displacement),
                    end_anchor_max_displacement_A=float(end_displacement),
                    start_connection_reference_length_A=left_reference,
                    end_connection_reference_length_A=right_reference,
                    start_connection_length_A=start_connection,
                    end_connection_length_A=end_connection,
                    score_components=score_components,
                    normalized_score=score,
                )
                total += 1
                candidates.append(candidate)
                candidates.sort(key=lambda item: (item.normalized_score, item.suite_ids))
                if len(candidates) > max_returned_candidates:
                    candidates.pop()
            if not complete:
                break

        status = "search_budget_exhausted" if not complete else "no_solution_found" if total == 0 else "complete"
        return DNAGapClosureResult(
            sequence=sequence,
            candidates=tuple(candidates),
            status=status,
            search_complete=complete,
            expanded_states=expanded,
            total_compatible_closures=total,
            candidates_truncated=total > len(candidates),
            start_frame_sha256=_frame_digest(start_A, coordinate_unit),
            end_frame_sha256=_frame_digest(end_A, coordinate_unit),
            start_anchor_atoms_sha256=_dna_anchor_digest(start_atoms),
            end_anchor_atoms_sha256=_dna_anchor_digest(end_atoms),
            source_files=tuple(copy.deepcopy(self.source_files)),
            limits={**positive_limits, **nonnegative_limits,
                    "max_search_states": max_search_states,
                    "max_returned_candidates": max_returned_candidates},
            rejections=rejections,
            trial_metric_minima=trial_metric_minima,
            trial_metric_evaluated_counts=trial_metric_evaluated_counts,
            assembly_rejection_reasons=assembly_rejection_reasons,
        )


def close_dna_gap_from_anchors(
    library: DNAFragmentLibrary,
    sequence: str,
    start_frame: Any,
    end_frame: Any,
    start_anchor_atoms: Mapping[str, Any],
    end_anchor_atoms: Mapping[str, Any],
    **options: Any,
) -> DNAGapClosureResult:
    """Close one internal DNA residue between two fixed anchor residues."""
    if not isinstance(library, DNAFragmentLibrary):
        raise TypeError("library must be a DNAFragmentLibrary.")
    return library.close_gap_from_anchors(
        sequence, start_frame, end_frame, start_anchor_atoms, end_anchor_atoms, **options
    )
