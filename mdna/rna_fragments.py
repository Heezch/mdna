"""Place and assemble source-traceable experimental RNA dinucleotide suites.

The library supports suite placement, bounded sequence path selection, and
assembly of covered single-strand segments from explicit base frames. The
public builders record provenance and hard limits. Coverage is finite; the
module does not infer frames from a centerline or build a complementary duplex.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation


_BASES = frozenset("ACGU")
_UNIT_TO_ANGSTROM = {"angstrom": 1.0, "nm": 10.0}
_FRAME_TOLERANCE = 1e-3


class UnsupportedSuiteSequenceError(ValueError):
    """Raised when no observed library fragment exists for a valid sequence."""


class NoCompatibleRNAFragmentPathError(ValueError):
    """Raised when no sequence of observed suites satisfies path tolerances."""


@dataclass(frozen=True)
class RNAFragmentPlacement:
    """Placed atom coordinates and the predicted downstream base frame."""

    suite_id: str
    sequence: str
    atoms: tuple[dict[str, Any], ...]
    bonds: tuple[dict[str, Any], ...]
    downstream_frame: np.ndarray
    translation_residual_A: float | None
    rotation_residual_degrees: float | None


@dataclass(frozen=True)
class RNAFragmentPathSelection:
    """Lowest-score suite chain satisfying explicit path and continuity limits.

    This records a fragment path candidate and its diagnostics. Atom
    coordinates for shared residues are not merged or relaxed here.
    """

    sequence: str
    placements: tuple[RNAFragmentPlacement, ...]
    shared_residue_rmsd_A: tuple[float, ...]
    normalized_score: float

    @property
    def suite_ids(self) -> tuple[str, ...]:
        return tuple(placement.suite_id for placement in self.placements)

    @property
    def max_translation_residual_A(self) -> float:
        return max(placement.translation_residual_A or 0.0 for placement in self.placements)

    @property
    def max_rotation_residual_degrees(self) -> float:
        return max(placement.rotation_residual_degrees or 0.0 for placement in self.placements)


@dataclass(frozen=True)
class RNAFragmentAssembly:
    """RNA segment with explicit terminal states and continuation ports."""

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
        """Convert this observed heavy-atom segment to a one-frame MDTraj object.

        Coordinates are converted from angstroms to MDTraj's nanometers. The
        topology records atom elements and covalent connectivity; source bond
        order labels remain available in ``self.bonds``. This segment keeps
        explicit heavy-atom bonds. Hydroxyl boundary states remain heavy-atom
        only; this method does not invent hydrogen coordinates.
        """
        import mdtraj as md

        sequence = self.sequence.upper() if isinstance(self.sequence, str) else ""
        if not sequence or set(sequence) - _BASES:
            raise ValueError("assembled segment sequence must contain standard RNA A/C/G/U only.")
        if len(sequence) != len(self.sequence):
            raise ValueError("assembled segment sequence length is invalid.")
        valid_five = {
            "open_phosphodiester_continuation": (0, "P"),
            "five_prime_hydroxyl": None,
        }
        valid_three = {
            "open_phosphodiester_continuation": (len(sequence) - 1, "O3'"),
            "three_prime_hydroxyl": None,
        }
        if self.five_prime_boundary_state not in valid_five or self.five_prime_port != valid_five[self.five_prime_boundary_state]:
            raise ValueError("5' boundary state and continuation port are inconsistent.")
        if self.three_prime_boundary_state not in valid_three or self.three_prime_port != valid_three[self.three_prime_boundary_state]:
            raise ValueError("3' boundary state and continuation port are inconsistent.")

        atom_records = {}
        coordinates_A = {}
        atom_elements = {}
        for record in self.atoms:
            residue_index = record.get("residue_index")
            atom_name = record.get("atom_name")
            element_symbol = str(record.get("element", "")).strip().upper()
            key = (residue_index, atom_name)
            if (not isinstance(residue_index, int) or residue_index < 0
                    or residue_index >= len(sequence) or not isinstance(atom_name, str)
                    or not atom_name.strip() or key in atom_records):
                raise ValueError(f"assembled segment has an invalid or duplicate atom record {key!r}.")
            try:
                element = md.element.get_by_symbol(element_symbol)
            except (KeyError, ValueError):
                element = None
            if element is None:
                raise ValueError(f"assembled atom {key!r} has unsupported element {element_symbol!r}.")
            xyz = np.asarray(record.get("xyz_A"), dtype=float)
            if xyz.shape != (3,) or not np.isfinite(xyz).all():
                raise ValueError(f"assembled atom {key!r} must have three finite coordinates in angstroms.")
            atom_records[key] = record
            atom_elements[key] = element
            coordinates_A[key] = xyz

        expected_residues = set(range(len(sequence)))
        observed_residues = {index for index, _ in atom_records}
        if observed_residues != expected_residues:
            raise ValueError("assembled atom records must cover every sequence residue in order.")
        for residue_index in range(len(sequence)):
            names = {name for index, name in atom_records if index == residue_index}
            missing = {"C1'", "O2'"} - names
            if missing:
                raise ValueError(
                    f"RNA residue {residue_index} lacks standard ribose atom(s): "
                    + ", ".join(sorted(missing)) + "."
                )
        if self.five_prime_boundary_state == "five_prime_hydroxyl":
            names = {name for index, name in atom_records if index == 0}
            if "O5'" not in names or names & {"P", "OP1", "OP2"}:
                raise ValueError("5' hydroxyl boundary must contain O5' and omit the phosphate group.")
        if self.three_prime_boundary_state == "three_prime_hydroxyl":
            names = {name for index, name in atom_records if index == len(sequence) - 1}
            if "O3'" not in names:
                raise ValueError("3' hydroxyl boundary must contain O3'.")
        if self.five_prime_port is not None and self.five_prime_port not in atom_records:
            raise ValueError("assembled segment is missing its declared 5' continuation atom.")
        if self.three_prime_port is not None and self.three_prime_port not in atom_records:
            raise ValueError("assembled segment is missing its declared 3' continuation atom.")

        topology = md.Topology()
        chain = topology.add_chain(chain_id="A")
        atom_index_by_key = {}
        for residue_index, base in enumerate(sequence):
            residue = topology.add_residue(base, chain, resSeq=residue_index + 1)
            for key in sorted(key for key in atom_records if key[0] == residue_index):
                atom_index_by_key[key] = topology.add_atom(
                    key[1], atom_elements[key], residue
                ).index

        seen_bonds = set()
        for bond in self.bonds:
            first = (bond.get("residue_index_1"), bond.get("atom_1"))
            second = (bond.get("residue_index_2"), bond.get("atom_2"))
            if first not in atom_index_by_key or second not in atom_index_by_key or first == second:
                raise ValueError(f"assembled bond has an invalid endpoint: {first!r}, {second!r}.")
            canonical = tuple(sorted((first, second)))
            if canonical in seen_bonds:
                raise ValueError(f"assembled segment contains duplicate bond {canonical!r}.")
            seen_bonds.add(canonical)
            topology.add_bond(
                topology.atom(atom_index_by_key[first]),
                topology.atom(atom_index_by_key[second]),
            )

        xyz_nm = np.stack([coordinates_A[key] for key in atom_index_by_key], axis=0) / 10.0
        return md.Trajectory(xyz_nm[np.newaxis, :, :], topology)

    def save_mmcif(self, filename: str | Path) -> Path:
        """Write the fragment's atom records and explicit covalent graph."""
        from .structure_export import save_fragment_mmcif

        return save_fragment_mmcif(self, filename, polymer_type="polyribonucleotide")

    def save_pdb(self, filename: str | Path) -> Path:
        """Write the internal segment with explicit CONECT records.

        MDTraj's PDB writer omits connectivity records. The method therefore
        writes the coordinates first, then adds the assembled atom graph so a
        PDB reader can recover phosphodiester links across residue boundaries.
        """
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


def _validated_frame(frame: Any, *, scale_to_angstrom: float, name: str) -> np.ndarray:
    array = np.asarray(frame, dtype=float)
    if array.shape != (4, 3):
        raise ValueError(f"{name} must have shape (4, 3): origin followed by three basis vectors.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite coordinates.")
    basis = array[1:]
    gram = basis @ basis.T
    if not np.allclose(gram, np.eye(3), atol=_FRAME_TOLERANCE, rtol=0.0):
        raise ValueError(f"{name} basis must be orthonormal.")
    if np.linalg.det(basis) <= 0.0:
        raise ValueError(f"{name} basis must be right-handed.")
    result = array.copy()
    result[0] *= scale_to_angstrom
    return result


def _validate_suite(record: Mapping[str, Any]) -> dict[str, Any]:
    suite = dict(record)
    suite_id = suite.get("suite_id")
    sequence = suite.get("sequence")
    if not isinstance(suite_id, str) or not suite_id.strip():
        raise ValueError("Each suite must have a non-empty suite_id.")
    if not isinstance(sequence, str) or len(sequence) != 2 or any(base not in _BASES for base in sequence):
        raise ValueError(f"Suite {suite_id!r} sequence must contain two standard RNA A/C/G/U bases.")
    if suite.get("coordinate_unit", "angstrom") != "angstrom":
        raise ValueError(f"Suite {suite_id!r} coordinates must use angstrom units.")
    residue_states = suite.get("residue_states")
    internal_state = "5prime_phosphodiester+3prime_phosphodiester"
    role = suite.get("terminal_role")
    expected_states = {
        None: [internal_state, internal_state],
        "five_prime": ["5prime_hydroxyl+3prime_phosphodiester", internal_state],
        "three_prime": [internal_state, "5prime_phosphodiester+3prime_hydroxyl"],
    }
    if role not in expected_states or residue_states != expected_states[role]:
        raise ValueError(f"Suite {suite_id!r} terminal role and residue states disagree.")

    atoms = suite.get("atoms")
    bonds = suite.get("bonds")
    if not isinstance(atoms, list) or not atoms:
        raise ValueError(f"Suite {suite_id!r} must contain atoms.")
    atom_keys: set[tuple[int, str]] = set()
    copied_atoms = []
    for atom in atoms:
        if not isinstance(atom, Mapping):
            raise ValueError(f"Suite {suite_id!r} has a malformed atom record.")
        index, atom_name = atom.get("residue_index"), atom.get("atom_name")
        if index not in (0, 1) or not isinstance(atom_name, str) or not atom_name:
            raise ValueError(f"Suite {suite_id!r} atom must identify residue 0/1 and an atom name.")
        key = (index, atom_name)
        if key in atom_keys:
            raise ValueError(f"Suite {suite_id!r} has duplicate atom {key!r}.")
        xyz = np.asarray(atom.get("xyz_local_A"), dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError(f"Suite {suite_id!r} atom {key!r} must have three finite local coordinates.")
        atom_keys.add(key)
        copied = dict(atom)
        copied["xyz_local_A"] = xyz.tolist()
        copied_atoms.append(copied)
    if (0, "O3'") not in atom_keys or (1, "P") not in atom_keys:
        raise ValueError(f"Suite {suite_id!r} must include upstream O3' and downstream P ports.")
    if role == "five_prime" and any((0, atom) in atom_keys for atom in ("P", "OP1", "OP2")):
        raise ValueError(f"Suite {suite_id!r} 5' hydroxyl residue cannot contain a phosphate group.")
    if role == "five_prime" and (0, "O5'") not in atom_keys:
        raise ValueError(f"Suite {suite_id!r} 5' hydroxyl residue must include O5'.")
    if role == "three_prime" and (1, "O3'") not in atom_keys:
        raise ValueError(f"Suite {suite_id!r} 3' hydroxyl residue must include O3'.")

    if not isinstance(bonds, list):
        raise ValueError(f"Suite {suite_id!r} must contain a bond list.")
    copied_bonds = []
    for bond in bonds:
        if not isinstance(bond, Mapping):
            raise ValueError(f"Suite {suite_id!r} has a malformed bond record.")
        first = (bond.get("residue_index_1"), bond.get("atom_1"))
        second = (bond.get("residue_index_2"), bond.get("atom_2"))
        if first not in atom_keys or second not in atom_keys:
            raise ValueError(f"Suite {suite_id!r} bond endpoint is absent: {first!r}, {second!r}.")
        if first == second:
            raise ValueError(f"Suite {suite_id!r} bond cannot connect an atom to itself.")
        copied_bonds.append(dict(bond))

    transform = suite.get("downstream_base_frame_relative_to_upstream")
    if not isinstance(transform, Mapping):
        raise ValueError(f"Suite {suite_id!r} lacks its downstream frame transform.")
    translation = np.asarray(transform.get("translation_A"), dtype=float)
    rotation = np.asarray(transform.get("rotation_matrix"), dtype=float)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError(f"Suite {suite_id!r} has an invalid downstream frame translation.")
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError(f"Suite {suite_id!r} has an invalid downstream frame rotation.")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=_FRAME_TOLERANCE, rtol=0.0):
        raise ValueError(f"Suite {suite_id!r} downstream frame rotation must be orthonormal.")
    if np.linalg.det(rotation) <= 0.0:
        raise ValueError(f"Suite {suite_id!r} downstream frame rotation must be right-handed.")

    suite["atoms"] = copied_atoms
    suite["bonds"] = copied_bonds
    suite["downstream_base_frame_relative_to_upstream"] = {
        "translation_A": translation.tolist(),
        "rotation_matrix": rotation.tolist(),
    }
    return suite


class RNAFragmentLibrary:
    """Validated observed RNA suite records with explicit sequence coverage."""

    def __init__(
        self,
        suites: list[Mapping[str, Any]],
        *,
        source_manifest_sha256: str | None = None,
        chemistry_registry_sha256: str | None = None,
    ):
        if not suites:
            raise ValueError("RNA fragment library must contain at least one suite.")
        validated = [_validate_suite(record) for record in suites]
        ids = [record["suite_id"] for record in validated]
        if len(ids) != len(set(ids)):
            raise ValueError("RNA fragment library contains duplicate suite_id values.")
        self._suites = tuple(validated)
        self.source_manifest_sha256 = source_manifest_sha256
        self.chemistry_registry_sha256 = chemistry_registry_sha256

    @classmethod
    def from_json(
        cls, path: str | Path, *, terminal_path: str | Path | None = None
    ) -> "RNAFragmentLibrary":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, Mapping) or data.get("schema_version") != 1:
            raise ValueError("Unsupported RNA fragment library schema_version; expected 1.")
        if data.get("coordinate_unit") != "angstrom":
            raise ValueError("RNA fragment library coordinate_unit must be angstrom.")
        for key in ("source_manifest_sha256", "chemistry_registry_sha256"):
            value = data.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"RNA fragment library requires a 64-character {key}.")
        suites = data.get("suites")
        if not isinstance(suites, list):
            raise ValueError("RNA fragment library suites must be a list.")
        if terminal_path is not None:
            with Path(terminal_path).open("r", encoding="utf-8") as handle:
                terminal_data = json.load(handle)
            if (
                not isinstance(terminal_data, Mapping)
                or terminal_data.get("schema_version") != 1
                or terminal_data.get("coordinate_unit") != "angstrom"
            or terminal_data.get("chemistry_registry_sha256") != data["chemistry_registry_sha256"]
            or terminal_data.get("source_manifest_sha256") != data["source_manifest_sha256"]
            ):
                raise ValueError("Terminal suite library schema, units, or chemistry registry do not match.")
            if not isinstance(terminal_data.get("suites"), list):
                raise ValueError("Terminal suite library suites must be a list.")
            suites = suites + terminal_data["suites"]
        return cls(
            suites,
            source_manifest_sha256=data["source_manifest_sha256"],
            chemistry_registry_sha256=data["chemistry_registry_sha256"],
        )

    @property
    def available_sequences(self) -> tuple[str, ...]:
        return tuple(sorted({
            suite["sequence"] for suite in self._suites
            if suite.get("terminal_role") is None
        }))

    def get_suites(
        self, sequence: str, *, terminal_role: str | None = None
    ) -> tuple[dict[str, Any], ...]:
        if not isinstance(sequence, str) or len(sequence) != 2 or any(base not in _BASES for base in sequence):
            raise ValueError("sequence must contain two standard RNA A/C/G/U bases.")
        if terminal_role not in {None, "five_prime", "three_prime"}:
            raise ValueError("terminal_role must be None, 'five_prime', or 'three_prime'.")
        matches = tuple(
            copy.deepcopy(suite) for suite in self._suites
            if suite["sequence"] == sequence and suite.get("terminal_role") == terminal_role
        )
        if not matches:
            covered = ", ".join(sorted({
                suite["sequence"] for suite in self._suites
                if suite.get("terminal_role") == terminal_role
            }))
            raise UnsupportedSuiteSequenceError(
                f"No observed RNA suite for {sequence} with terminal_role={terminal_role!r}; "
                f"current library coverage: {covered}."
            )
        return matches

    def place_suite(
        self,
        suite: Mapping[str, Any],
        upstream_frame: Any,
        *,
        coordinate_unit: str,
        downstream_frame: Any | None = None,
        balance_frames: bool = False,
    ) -> RNAFragmentPlacement:
        """Place one suite; target downstream frame, if supplied, is in angstrom.

        ``upstream_frame`` uses the MDNA row-basis convention: origin followed
        by right-handed x/y/z basis vectors. The upstream origin uses
        ``coordinate_unit``; all output coordinates and residuals are in Å.
        """
        if coordinate_unit not in _UNIT_TO_ANGSTROM:
            raise ValueError("coordinate_unit must be 'angstrom' or 'nm'.")
        validated_suite = _validate_suite(suite)
        if validated_suite not in self._suites:
            raise ValueError("suite must be a record obtained from this library.")
        upstream = _validated_frame(
            upstream_frame,
            scale_to_angstrom=_UNIT_TO_ANGSTROM[coordinate_unit],
            name="upstream_frame",
        )
        target = None if downstream_frame is None else _validated_frame(
            downstream_frame, scale_to_angstrom=1.0, name="downstream_frame"
        )
        if not isinstance(balance_frames, bool) or (balance_frames and target is None):
            raise ValueError("balance_frames requires a boolean and a downstream frame.")

        origin = upstream[0]
        basis_rows = upstream[1:]
        transformed_atoms = []
        for atom in validated_suite["atoms"]:
            xyz_A = np.asarray(atom["xyz_local_A"], dtype=float) @ basis_rows + origin
            placed = {key: value for key, value in atom.items() if key != "xyz_local_A"}
            placed["xyz_A"] = xyz_A.tolist()
            transformed_atoms.append(placed)

        relative = validated_suite["downstream_base_frame_relative_to_upstream"]
        predicted_basis_rows = np.asarray(relative["rotation_matrix"], dtype=float).T @ basis_rows
        predicted_origin = origin + np.asarray(relative["translation_A"], dtype=float) @ basis_rows
        predicted_frame = np.vstack((predicted_origin, predicted_basis_rows))

        translation_residual = rotation_residual = None
        if target is not None:
            if balance_frames:
                correction = Rotation.from_matrix(target[1:].T @ predicted_basis_rows)
                half = Rotation.from_rotvec(.5 * correction.as_rotvec())
                # Center the rigid correction between both residue origins.
                pivot = .5 * (origin + predicted_origin)
                shift = .5 * (target[0] - predicted_origin)
                def move(point):
                    return half.apply(point - pivot) + pivot + shift

                for atom in transformed_atoms:
                    atom["xyz_A"] = move(np.asarray(atom["xyz_A"])).tolist()
                predicted_origin = move(predicted_origin)
                predicted_basis_rows = half.apply(predicted_basis_rows)
                predicted_frame = np.vstack((predicted_origin, predicted_basis_rows))
            translation_residual = float(np.linalg.norm(predicted_origin - target[0]))
            relative_orientation = predicted_basis_rows @ target[1:].T
            cosine = np.clip((np.trace(relative_orientation) - 1.0) / 2.0, -1.0, 1.0)
            rotation_residual = float(np.degrees(np.arccos(cosine)))
            if balance_frames:
                translation_residual = max(translation_residual, float(np.linalg.norm(move(origin)-origin)))
                rotation_residual = max(rotation_residual, float(np.degrees(half.magnitude())))

        return RNAFragmentPlacement(
            suite_id=validated_suite["suite_id"],
            sequence=validated_suite["sequence"],
            atoms=tuple(transformed_atoms),
            bonds=tuple(dict(bond) for bond in validated_suite["bonds"]),
            downstream_frame=predicted_frame,
            translation_residual_A=translation_residual,
            rotation_residual_degrees=rotation_residual,
        )

    @staticmethod
    def _shared_residue_rmsd_A(
        first: RNAFragmentPlacement,
        second: RNAFragmentPlacement,
    ) -> float | None:
        """Compare at least three common heavy atoms in the shared residue."""
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
        differences = np.stack([first_atoms[name] - second_atoms[name] for name in names])
        return float(np.sqrt(np.mean(np.sum(differences * differences, axis=1))))

    def select_suite_path(
        self,
        sequence: str,
        base_frames: Any,
        *,
        coordinate_unit: str,
        max_translation_residual_A: float,
        max_rotation_residual_degrees: float,
        max_shared_residue_rmsd_A: float,
        balance_frames: bool = False,
        include_terminal_suites: bool = False,
        assembly_limits: tuple[float, float] | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> RNAFragmentPathSelection:
        """Select a compatible suite chain for caller-supplied nucleotide frames.

        Every sequence dinucleotide must be represented in the library. Frame
        residuals and shared-residue RMSDs are hard limits set by the caller;
        normalized squared residuals determine the lowest-score compatible
        chain. Optional terminal suites are required at both ends and internal
        suites are required between them. Ties are resolved by suite IDs for reproducibility. The returned
        fragments remain separate records: this method does not synthesize a
        single coordinate set, optimize torsions, or close terminal anchors.
        If supplied, diagnostics receives per-step gate counts and failure stage.
        """
        if not isinstance(sequence, str) or len(sequence) < 2 or any(base not in _BASES for base in sequence):
            raise ValueError("sequence must contain at least two standard RNA A/C/G/U bases.")
        if include_terminal_suites and len(sequence) < 3:
            raise ValueError("terminal suite paths require at least three residues.")
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

        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update(frame_steps=[], joins=[], failure=None)
        candidates_by_step: list[list[RNAFragmentPlacement]] = []
        for step in range(len(sequence) - 1):
            pair = sequence[step:step + 2]
            role = None
            if include_terminal_suites and step == 0:
                role = "five_prime"
            elif include_terminal_suites and step == len(sequence) - 2:
                role = "three_prime"
            try:
                suite_records = sorted(
                    self.get_suites(pair, terminal_role=role), key=lambda item: item["suite_id"]
                )
            except UnsupportedSuiteSequenceError:
                if diagnostics is not None:
                    diagnostics["failure"] = dict(stage="coverage", step=step, pair=pair, role=role)
                raise
            candidates = []
            step_metrics = dict(step=step, pair=pair, templates=len(suite_records),
                                translation_only_failed=0, rotation_only_failed=0, both_failed=0,
                                passed=0, closest=None)
            closest = None
            for suite in suite_records:
                placement = self.place_suite(
                    suite,
                    frames_A[step],
                    coordinate_unit="angstrom",
                    downstream_frame=frames_A[step + 1],
                    balance_frames=balance_frames,
                )
                t = placement.translation_residual_A
                r = placement.rotation_residual_degrees
                t_ok, r_ok = t <= max_translation_residual_A, r <= max_rotation_residual_degrees
                if diagnostics is not None:
                    key = ("passed" if t_ok and r_ok else "both_failed" if not t_ok and not r_ok
                           else "translation_only_failed" if not t_ok else "rotation_only_failed")
                    step_metrics[key] += 1
                    rank = (max(t/max_translation_residual_A, r/max_rotation_residual_degrees), suite["suite_id"])
                    if closest is None or rank < closest:
                        closest = rank
                        step_metrics["closest"] = dict(suite_id=rank[1], translation_A=t, rotation_deg=r,
                                                       max_limit_ratio=rank[0])
                if t_ok and r_ok:
                    candidates.append(placement)
            if diagnostics is not None:
                diagnostics["frame_steps"].append(step_metrics)
            if not candidates:
                if diagnostics is not None:
                    diagnostics["failure"] = dict(stage="frame_residual", step=step, pair=pair)
                raise NoCompatibleRNAFragmentPathError(
                    f"No suite at step {step} ({pair}) satisfies the "
                    f"{'two-end' if balance_frames else 'downstream'} frame residual limits."
                )
            candidates_by_step.append(candidates)

        # Dynamic programming keeps the best compatible prefix ending at each
        # suite, rather than enumerating the Cartesian product of conformers.
        states: list[dict[str, Any]] = []
        for candidate in candidates_by_step[0]:
            translation = candidate.translation_residual_A / max_translation_residual_A
            rotation = candidate.rotation_residual_degrees / max_rotation_residual_degrees
            states.append({"cost": translation * translation + rotation * rotation,
                           "path": (candidate.suite_id,), "previous": None,
                           "placement": candidate, "shared": None})

        for step in range(1, len(candidates_by_step)):
            next_states = []
            join = dict(step=step, attempted=0, continuity_rejected=0, assembly_rejected=0,
                        passed=0, best_shared_rmsd_A=None, first_assembly_error=None)
            for candidate in candidates_by_step[step]:
                best = None
                translation = candidate.translation_residual_A / max_translation_residual_A
                rotation = candidate.rotation_residual_degrees / max_rotation_residual_degrees
                candidate_cost = translation * translation + rotation * rotation
                for previous_index, previous in enumerate(states):
                    shared = self._shared_residue_rmsd_A(previous["placement"], candidate)
                    if diagnostics is not None:
                        join["attempted"] += 1
                        if shared is not None and (join["best_shared_rmsd_A"] is None
                                                   or shared < join["best_shared_rmsd_A"]):
                            join["best_shared_rmsd_A"] = shared
                    if shared is None or shared > max_shared_residue_rmsd_A:
                        if diagnostics is not None:
                            join["continuity_rejected"] += 1
                        continue
                    if assembly_limits is not None:
                        pair_path = RNAFragmentPathSelection(
                            sequence=sequence[step-1:step+2],
                            placements=(previous["placement"], candidate),
                            shared_residue_rmsd_A=(shared,), normalized_score=0.)
                        try:
                            self.assemble_suite_path(
                                pair_path, max_shared_atom_displacement_A=assembly_limits[0],
                                max_bond_length_deviation_A=assembly_limits[1])
                        except ValueError as exc:
                            if diagnostics is not None:
                                join["assembly_rejected"] += 1
                                if join["first_assembly_error"] is None:
                                    join["first_assembly_error"] = str(exc)
                            continue
                    if diagnostics is not None:
                        join["passed"] += 1
                    continuity = shared / max_shared_residue_rmsd_A
                    option = {
                        "cost": previous["cost"] + candidate_cost + continuity * continuity,
                        "path": previous["path"] + (candidate.suite_id,),
                        "previous": previous_index,
                        "placement": candidate,
                        "shared": shared,
                    }
                    if best is None or (option["cost"], option["path"]) < (best["cost"], best["path"]):
                        best = option
                if best is not None:
                    next_states.append(best)
            if diagnostics is not None:
                diagnostics["joins"].append(join)
            if not next_states:
                if diagnostics is not None:
                    diagnostics["failure"] = dict(stage="join", step=step, pair=sequence[step:step+2])
                raise NoCompatibleRNAFragmentPathError(
                    f"No adjacent suite pair at step {step} ({sequence[step:step + 2]}) "
                    + ("satisfies the continuity and assembly limits." if assembly_limits is not None
                       else "satisfies the shared-residue continuity limit.")
                )
            states = next_states

        best_path = min(states, key=lambda state: (state["cost"], state["path"]))
        # Reconstruct placements along the selected dynamic-programming path.
        selected_ids = best_path["path"]
        selected = []
        for step, suite_id in enumerate(selected_ids):
            selected.append(next(item for item in candidates_by_step[step]
                                 if item.suite_id == suite_id))
        shared_rmsd = tuple(
            self._shared_residue_rmsd_A(selected[index], selected[index + 1])
            for index in range(len(selected) - 1)
        )
        return RNAFragmentPathSelection(
            sequence=sequence,
            placements=tuple(selected),
            shared_residue_rmsd_A=tuple(float(value) for value in shared_rmsd),
            normalized_score=float(best_path["cost"]),
        )

    def assemble_suite_path(
        self,
        path: RNAFragmentPathSelection,
        *,
        max_shared_atom_displacement_A: float,
        max_bond_length_deviation_A: float,
        shared_coordinates: Mapping[int, Mapping[str, Any]] | None = None,
    ) -> RNAFragmentAssembly:
        """Join a selected suite path only when shared coordinates agree.

        The first copy of a shared atom is retained. No coordinate averaging,
        torsion fitting, proton placement, or endpoint closure is performed.
        Bond-length deviations are measured against every source occurrence
        of that bond, so this checks assembly distortion rather than absolute
        chemical geometry quality.
        Optional explicit shared coordinates are checked against both source
        copies and every original source bond.
        """
        if not isinstance(path, RNAFragmentPathSelection):
            raise TypeError("path must be an RNAFragmentPathSelection.")
        if len(path.placements) != len(path.sequence) - 1 or len(path.sequence) < 2:
            raise ValueError("path must contain one suite placement per sequence dinucleotide.")
        limits = {
            "max_shared_atom_displacement_A": max_shared_atom_displacement_A,
            "max_bond_length_deviation_A": max_bond_length_deviation_A,
        }
        for name, value in limits.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number.")

        suite_by_id = {suite["suite_id"]: suite for suite in self._suites}
        for index, placement in enumerate(path.placements):
            source = suite_by_id.get(placement.suite_id)
            if source is None or source["sequence"] != path.sequence[index:index + 2]:
                raise ValueError(f"path suite at step {index} is not registered for its sequence.")
            role = source.get("terminal_role")
            expected_role = (
                "five_prime" if index == 0 else
                "three_prime" if index == len(path.placements) - 1 else None
            )
            if role not in (None, expected_role):
                raise ValueError(f"path suite at step {index} has terminal role {role!r} in an invalid position.")
            expected_atoms = {
                (atom["residue_index"], atom["atom_name"]) for atom in source["atoms"]
            }
            actual_atoms = {
                (atom.get("residue_index"), atom.get("atom_name")) for atom in placement.atoms
            }
            if actual_atoms != expected_atoms:
                raise ValueError(f"path suite at step {index} atom inventory differs from its library record.")
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
            if actual_bonds != expected_bonds:
                raise ValueError(f"path suite at step {index} bond graph differs from its library record.")

        overrides = shared_coordinates or {}
        if any(not isinstance(i, int) or not 0 < i < len(path.sequence)-1 for i in overrides):
            raise ValueError("shared coordinates may only replace overlap residues.")
        atom_records: dict[tuple[int, str], dict[str, Any]] = {}
        residue_inventories: dict[int, dict[str, str]] = {}
        shared_differences: dict[int, list[float]] = {}
        bond_records: dict[tuple[tuple[int, str], tuple[int, str]], dict[str, Any]] = {}
        bond_reference_lengths: list[tuple[tuple[tuple[int, str], tuple[int, str]], float]] = []

        for step, placement in enumerate(path.placements):
            local_atoms = {}
            local_residue_inventories: dict[int, dict[str, str]] = {0: {}, 1: {}}
            for atom in placement.atoms:
                local_index = atom.get("residue_index")
                name = atom.get("atom_name")
                element = atom.get("element", "")
                xyz = np.asarray(atom.get("xyz_A"), dtype=float)
                if local_index not in (0, 1) or not isinstance(name, str) or not name:
                    raise ValueError(f"suite {placement.suite_id!r} has a malformed placed atom.")
                if xyz.shape != (3,) or not np.isfinite(xyz).all():
                    raise ValueError(f"suite {placement.suite_id!r} contains invalid placed coordinates.")
                key_local = (local_index, name)
                local_atoms[key_local] = (atom, xyz)
                local_residue_inventories[local_index][name] = str(element).upper()
                global_key = (step + local_index, name)
                copied = dict(atom)
                copied["residue_index"] = step + local_index
                copied["xyz_A"] = xyz.tolist()
                override = overrides.get(global_key[0])
                if override is not None:
                    if name not in override:
                        raise ValueError(f"shared coordinates omit atom {global_key!r}.")
                    supplied = np.asarray(override[name], dtype=float)
                    if supplied.shape != (3,) or not np.isfinite(supplied).all():
                        raise ValueError(f"invalid shared coordinates for atom {global_key!r}.")
                    displacement = float(np.linalg.norm(supplied - xyz))
                    if displacement > max_shared_atom_displacement_A:
                        raise ValueError("shared coordinates exceed source displacement limit.")
                    shared_differences.setdefault(global_key[0], []).append(displacement)
                    copied["xyz_A"] = supplied.tolist()
                existing = atom_records.get(global_key)
                if existing is None:
                    atom_records[global_key] = copied
                else:
                    if existing.get("element", "").upper() != str(element).upper():
                        raise ValueError(f"shared atom {global_key!r} has conflicting elements.")
                    if override is not None:
                        continue
                    displacement = float(np.linalg.norm(np.asarray(existing["xyz_A"]) - xyz))
                    shared_differences.setdefault(step, []).append(displacement)
                    if displacement > max_shared_atom_displacement_A:
                        raise ValueError(
                            f"shared atom displacement at residue {step} is {displacement:.6g} Å, "
                            f"above limit {max_shared_atom_displacement_A:.6g} Å."
                        )

            for local_index, inventory in local_residue_inventories.items():
                global_index = step + local_index
                if global_index in overrides and set(overrides[global_index]) != inventory.keys():
                    raise ValueError("shared coordinate inventory differs from the source.")
                previous_inventory = residue_inventories.get(global_index)
                if previous_inventory is not None and previous_inventory != inventory:
                    raise ValueError(
                        f"shared residue {global_index} atom inventory or elements differ between suites."
                    )
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
                if existing_bond is not None and existing_bond.get("order") != order:
                    raise ValueError(f"bond {canonical!r} has conflicting bond orders.")
                if existing_bond is None:
                    bond_records[canonical] = {
                        "residue_index_1": first_global[0], "atom_1": first_global[1],
                        "residue_index_2": second_global[0], "atom_2": second_global[1],
                        "order": order,
                    }
                first_xyz = local_atoms[first_local][1]
                second_xyz = local_atoms[second_local][1]
                bond_reference_lengths.append((canonical, float(np.linalg.norm(first_xyz - second_xyz))))

        for canonical in bond_records:
            if any(key not in atom_records for key in canonical):
                raise ValueError(f"assembled bond endpoint is absent: {canonical!r}.")
        first_suite = suite_by_id[path.placements[0].suite_id]
        last_suite = suite_by_id[path.placements[-1].suite_id]
        five_terminal = first_suite.get("terminal_role") == "five_prime"
        three_terminal = last_suite.get("terminal_role") == "three_prime"
        five_prime_port = None if five_terminal else (0, "P")
        three_prime_port = None if three_terminal else (len(path.sequence) - 1, "O3'")
        if five_prime_port is not None and five_prime_port not in atom_records:
            raise ValueError("assembled segment lacks its 5' P continuation port.")
        if three_prime_port is not None and three_prime_port not in atom_records:
            raise ValueError("assembled segment lacks its 3' O3' continuation port.")
        bond_deviations = []
        for canonical, reference_length in bond_reference_lengths:
            actual_length = np.linalg.norm(
                np.asarray(atom_records[canonical[0]]["xyz_A"])
                - np.asarray(atom_records[canonical[1]]["xyz_A"])
            )
            deviation = float(abs(actual_length - reference_length))
            bond_deviations.append(deviation)
            if deviation > max_bond_length_deviation_A:
                raise ValueError(
                    f"assembled bond length changed by {deviation:.6g} Å, above limit "
                    f"{max_bond_length_deviation_A:.6g} Å; endpoints {canonical!r}."
                )

        interface_rmsd = tuple(
            float(np.sqrt(np.mean(np.square(shared_differences[index]))))
            for index in range(1, len(path.placements))
        )
        return RNAFragmentAssembly(
            sequence=path.sequence,
            atoms=tuple(atom_records[key] for key in sorted(atom_records)),
            bonds=tuple(bond_records[key] for key in sorted(bond_records)),
            source_suite_ids=tuple(placement.suite_id for placement in path.placements),
            five_prime_boundary_state=(
                "five_prime_hydroxyl" if five_terminal else "open_phosphodiester_continuation"
            ),
            five_prime_port=five_prime_port,
            three_prime_boundary_state=(
                "three_prime_hydroxyl" if three_terminal else "open_phosphodiester_continuation"
            ),
            three_prime_port=three_prime_port,
            shared_residue_rmsd_A=interface_rmsd,
            shared_residue_max_displacement_A=max(
                (value for values in shared_differences.values() for value in values), default=0.0
            ),
            max_bond_length_deviation_A=max(bond_deviations, default=0.0),
        )


@dataclass(frozen=True)
class RNAChainBuildResult:
    """Constructed observed-coordinate RNA segment and reproducibility report."""

    path_selection: RNAFragmentPathSelection
    assembly: RNAFragmentAssembly
    coordinate_unit: str
    target_base_frames_sha256: str
    source_manifest_sha256: str | None
    chemistry_registry_sha256: str | None
    max_translation_residual_A: float
    max_rotation_residual_degrees: float
    max_shared_residue_rmsd_A: float
    max_shared_atom_displacement_A: float
    max_bond_length_deviation_A: float
    include_terminal_suites: bool
    balance_frames: bool = False

    def to_report(self) -> dict[str, Any]:
        """Return JSON-compatible construction parameters and diagnostics."""
        return {
            "schema_version": 1,
            "algorithm": "rna-suite-dp-hard-limits-v1",
            "sequence": self.path_selection.sequence,
            "coordinate_unit": self.coordinate_unit,
            "target_base_frames_sha256": self.target_base_frames_sha256,
            "source_manifest_sha256": self.source_manifest_sha256,
            "chemistry_registry_sha256": self.chemistry_registry_sha256,
            "include_terminal_suites": self.include_terminal_suites,
            "balance_frames": self.balance_frames,
            "limits": {
                "max_translation_residual_A": self.max_translation_residual_A,
                "max_rotation_residual_degrees": self.max_rotation_residual_degrees,
                "max_shared_residue_rmsd_A": self.max_shared_residue_rmsd_A,
                "max_shared_atom_displacement_A": self.max_shared_atom_displacement_A,
                "max_bond_length_deviation_A": self.max_bond_length_deviation_A,
            },
            "suite_ids": list(self.path_selection.suite_ids),
            "selection": {
                "normalized_score": self.path_selection.normalized_score,
                "shared_residue_rmsd_A": list(self.path_selection.shared_residue_rmsd_A),
                "max_translation_residual_A": self.path_selection.max_translation_residual_A,
                "max_rotation_residual_degrees": self.path_selection.max_rotation_residual_degrees,
            },
            "assembly": {
                "atom_count": len(self.assembly.atoms),
                "bond_count": len(self.assembly.bonds),
                "five_prime_boundary_state": self.assembly.five_prime_boundary_state,
                "five_prime_port": self.assembly.five_prime_port,
                "three_prime_boundary_state": self.assembly.three_prime_boundary_state,
                "three_prime_port": self.assembly.three_prime_port,
                "shared_residue_max_displacement_A": (
                    self.assembly.shared_residue_max_displacement_A
                ),
                "max_bond_length_deviation_A": self.assembly.max_bond_length_deviation_A,
            },
        }


@dataclass(frozen=True)
class RNAGapClosureCandidate:
    """One assembled internal RNA segment that satisfies both fixed anchors."""

    path_selection: RNAFragmentPathSelection
    assembly: RNAFragmentAssembly
    endpoint_translation_residual_A: float
    endpoint_rotation_residual_degrees: float
    start_anchor_max_displacement_A: float
    end_anchor_max_displacement_A: float
    start_anchor_connection_reference_length_A: float
    end_anchor_connection_reference_length_A: float
    start_anchor_connection_length_A: float
    end_anchor_connection_length_A: float
    start_anchor_connection_length_deviation_A: float
    end_anchor_connection_length_deviation_A: float
    environment_min_distance_A: float | None
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
            "start_anchor_connection_reference_length_A": (
                self.start_anchor_connection_reference_length_A
            ),
            "end_anchor_connection_reference_length_A": (
                self.end_anchor_connection_reference_length_A
            ),
            "start_anchor_connection_length_A": self.start_anchor_connection_length_A,
            "end_anchor_connection_length_A": self.end_anchor_connection_length_A,
            "start_anchor_connection_length_deviation_A": (
                self.start_anchor_connection_length_deviation_A
            ),
            "end_anchor_connection_length_deviation_A": (
                self.end_anchor_connection_length_deviation_A
            ),
            "environment_min_distance_A": self.environment_min_distance_A,
            "score_components": dict(self.score_components),
            "normalized_score": self.normalized_score,
            "assembly_atom_count": len(self.assembly.atoms),
            "assembly_bond_count": len(self.assembly.bonds),
            "assembly_shared_atom_max_displacement_A": (
                self.assembly.shared_residue_max_displacement_A
            ),
            "assembly_max_bond_length_deviation_A": self.assembly.max_bond_length_deviation_A,
        }


@dataclass(frozen=True)
class RNAGapClosureResult:
    """Bounded search outcome; a candidate is software-feasible, not validated biology."""

    sequence: str
    candidates: tuple[RNAGapClosureCandidate, ...]
    status: str
    search_complete: bool
    expanded_states: int
    total_compatible_closures: int
    candidates_truncated: bool
    start_frame_sha256: str
    end_frame_sha256: str
    start_anchor_atoms_sha256: str
    end_anchor_atoms_sha256: str
    source_manifest_sha256: str | None
    chemistry_registry_sha256: str | None
    environment_atoms_sha256: str
    limits: Mapping[str, float | int | None]
    rejections: Mapping[str, int]
    anchors_pinned: bool = False
    distinct_missing_geometries: int | None = None
    anchor_angle_windows_deg: Mapping[str, tuple[float, float]] | None = None

    @property
    def missing_sequence(self) -> str:
        return self.sequence[1:-1]

    @property
    def missing_residue_indices(self) -> tuple[int, ...]:
        return tuple(range(1, len(self.sequence) - 1))

    def to_report(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "algorithm": "rna-suite-anchor-dfs-hard-limits-v2",
            "anchors_pinned": self.anchors_pinned,
            "distinct_missing_geometries": self.distinct_missing_geometries,
            "anchor_angle_windows_deg": dict(self.anchor_angle_windows_deg)
            if self.anchor_angle_windows_deg is not None else None,
            "sequence_including_anchors": self.sequence,
            "missing_sequence": self.missing_sequence,
            "missing_residue_indices": list(self.missing_residue_indices),
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
            "source_manifest_sha256": self.source_manifest_sha256,
            "chemistry_registry_sha256": self.chemistry_registry_sha256,
            "environment_atoms_sha256": self.environment_atoms_sha256,
            "limits": dict(self.limits),
            "rejections": dict(self.rejections),
            "candidates": [candidate.to_report() for candidate in self.candidates],
            "scientific_acceptance": False,
        }


def build_rna_chain_from_frames(
    library: RNAFragmentLibrary,
    sequence: str,
    base_frames: Any,
    *,
    coordinate_unit: str,
    max_translation_residual_A: float,
    max_rotation_residual_degrees: float,
    max_shared_residue_rmsd_A: float,
    max_shared_atom_displacement_A: float,
    max_bond_length_deviation_A: float,
    include_terminal_suites: bool,
    balance_frames: bool = False,
    diagnostics: dict[str, Any] | None = None,
) -> RNAChainBuildResult:
    """Select and assemble a covered RNA segment from explicit base frames.

    Every limit and the decision to require chemically capped terminal
    fragments are explicit inputs. Unsupported sequence coverage, invalid
    frames, incompatible adjacent fragments, or assembly distortion is
    reported as an exception. The result retains its source IDs and can emit a
    JSON-compatible diagnostics report; no coordinates are relaxed or invented.
    """
    if not isinstance(library, RNAFragmentLibrary):
        raise TypeError("library must be an RNAFragmentLibrary.")
    if not isinstance(include_terminal_suites, bool):
        raise ValueError("include_terminal_suites must be a bool.")
    selection = library.select_suite_path(
        sequence,
        base_frames,
        coordinate_unit=coordinate_unit,
        max_translation_residual_A=max_translation_residual_A,
        max_rotation_residual_degrees=max_rotation_residual_degrees,
        max_shared_residue_rmsd_A=max_shared_residue_rmsd_A,
        include_terminal_suites=include_terminal_suites,
        balance_frames=balance_frames,
        diagnostics=diagnostics,
        assembly_limits=(max_shared_atom_displacement_A, max_bond_length_deviation_A)
        if balance_frames else None,
    )
    try:
        assembly = library.assemble_suite_path(
            selection,
            max_shared_atom_displacement_A=max_shared_atom_displacement_A,
            max_bond_length_deviation_A=max_bond_length_deviation_A,
        )
    except ValueError as exc:
        if diagnostics is not None:
            diagnostics["failure"] = dict(stage="final_assembly", reason=str(exc))
        raise
    frames = np.asarray(base_frames, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(coordinate_unit.encode("ascii"))
    digest.update(np.asarray(frames.shape, dtype="<i8").tobytes())
    digest.update(frames.tobytes(order="C"))
    return RNAChainBuildResult(
        path_selection=selection,
        assembly=assembly,
        coordinate_unit=coordinate_unit,
        target_base_frames_sha256=digest.hexdigest(),
        source_manifest_sha256=library.source_manifest_sha256,
        chemistry_registry_sha256=library.chemistry_registry_sha256,
        max_translation_residual_A=max_translation_residual_A,
        max_rotation_residual_degrees=max_rotation_residual_degrees,
        max_shared_residue_rmsd_A=max_shared_residue_rmsd_A,
        max_shared_atom_displacement_A=max_shared_atom_displacement_A,
        max_bond_length_deviation_A=max_bond_length_deviation_A,
        include_terminal_suites=include_terminal_suites,
        balance_frames=balance_frames,
    )


def _validate_anchor_atom_map(anchor_atoms: Mapping[str, Any], *, name: str) -> dict[str, np.ndarray]:
    if not isinstance(anchor_atoms, Mapping) or len(anchor_atoms) < 3:
        raise ValueError(f"{name} must map at least three anchor atom names to coordinates.")
    validated = {}
    for atom_name, coordinates in anchor_atoms.items():
        if not isinstance(atom_name, str) or not atom_name.strip():
            raise ValueError(f"{name} contains an empty or invalid atom name.")
        xyz = np.asarray(coordinates, dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError(f"{name}[{atom_name!r}] must contain three finite coordinates in angstroms.")
        validated[atom_name] = xyz
    return validated


def _anchor_displacement(
    placement: RNAFragmentPlacement,
    local_residue_index: int,
    anchor_atoms: Mapping[str, np.ndarray],
) -> float | None:
    placed_atoms = {
        atom["atom_name"]: np.asarray(atom["xyz_A"], dtype=float)
        for atom in placement.atoms
        if atom["residue_index"] == local_residue_index
    }
    if not anchor_atoms.keys() <= placed_atoms.keys():
        return None
    return max(
        float(np.linalg.norm(placed_atoms[name] - xyz))
        for name, xyz in anchor_atoms.items()
    )


def _source_o3p_bond_length(suite: Mapping[str, Any]) -> float | None:
    atoms = {
        (atom["residue_index"], atom["atom_name"]): np.asarray(atom["xyz_local_A"], dtype=float)
        for atom in suite["atoms"]
    }
    for bond in suite["bonds"]:
        first = (bond["residue_index_1"], bond["atom_1"])
        second = (bond["residue_index_2"], bond["atom_2"])
        if {first, second} == {(0, "O3'"), (1, "P")}:
            return float(np.linalg.norm(atoms[first] - atoms[second]))
    return None


def _frame_digest(frame: np.ndarray, coordinate_unit: str) -> str:
    digest = hashlib.sha256()
    digest.update(coordinate_unit.encode("ascii"))
    digest.update(np.asarray(frame.shape, dtype="<i8").tobytes())
    digest.update(np.asarray(frame, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def _anchor_atoms_digest(anchor_atoms: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for atom_name in sorted(anchor_atoms):
        digest.update(atom_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.asarray(anchor_atoms[atom_name], dtype="<f8").tobytes())
    return digest.hexdigest()


def _validate_environment_atoms(
    environment_atoms: Any,
    min_environment_distance_A: Any,
    excluded_environment_pairs: Any,
    sequence: str,
) -> tuple[tuple[dict[str, Any], ...], frozenset[tuple[int, str, str]], str]:
    if not isinstance(excluded_environment_pairs, (list, tuple, set, frozenset)):
        raise ValueError(
            "excluded_environment_pairs must be a collection of (residue_index, atom_name, atom_id) triples."
        )
    if environment_atoms is None:
        if min_environment_distance_A is not None or len(excluded_environment_pairs):
            raise ValueError(
                "environment_atoms must be supplied when a distance cutoff or exclusions are set."
            )
        atoms: tuple[dict[str, Any], ...] = ()
    else:
        if (isinstance(min_environment_distance_A, bool)
                or not isinstance(min_environment_distance_A, (int, float, np.integer, np.floating))
                or not np.isfinite(min_environment_distance_A)
                or min_environment_distance_A <= 0):
            raise ValueError(
                "min_environment_distance_A must be positive and finite when environment_atoms are supplied."
            )
        if not isinstance(environment_atoms, (list, tuple)):
            raise ValueError("environment_atoms must be a list or tuple of heavy-atom records.")
        parsed = []
        seen_ids = set()
        for index, record in enumerate(environment_atoms):
            if not isinstance(record, Mapping):
                raise ValueError(f"environment_atoms[{index}] must be a mapping.")
            atom_id = record.get("atom_id")
            element = str(record.get("element", "")).strip().upper()
            xyz = np.asarray(record.get("xyz_A"), dtype=float)
            if not isinstance(atom_id, str) or not atom_id.strip() or atom_id in seen_ids:
                raise ValueError("environment atom IDs must be unique non-empty strings.")
            if not element or element in {"H", "D"}:
                raise ValueError(f"environment atom {atom_id!r} must have a non-hydrogen element.")
            if xyz.shape != (3,) or not np.isfinite(xyz).all():
                raise ValueError(f"environment atom {atom_id!r} must have three finite xyz_A coordinates.")
            seen_ids.add(atom_id)
            parsed.append({"atom_id": atom_id, "element": element, "xyz_A": xyz})
        atoms = tuple(parsed)

    atom_ids = {atom["atom_id"] for atom in atoms}
    exclusions = set()
    for pair in excluded_environment_pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 3:
            raise ValueError(
                "each excluded environment pair must be a (residue_index, atom_name, atom_id) triple."
            )
        residue_index, atom_name, atom_id = pair
        if (isinstance(residue_index, bool) or not isinstance(residue_index, int)
                or not 1 <= residue_index < len(sequence) - 1
                or not isinstance(atom_name, str) or not atom_name.strip()
                or not isinstance(atom_id, str) or atom_id not in atom_ids):
            raise ValueError(f"invalid excluded environment pair {pair!r}.")
        exclusions.add((residue_index, atom_name, atom_id))

    digest = hashlib.sha256()
    for atom in sorted(atoms, key=lambda item: item["atom_id"]):
        digest.update(atom["atom_id"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(atom["element"].encode("ascii"))
        digest.update(b"\0")
        digest.update(np.asarray(atom["xyz_A"], dtype="<f8").tobytes())
    for pair in sorted(exclusions):
        digest.update(repr(pair).encode("utf-8"))
        digest.update(b"\0")
    return atoms, frozenset(exclusions), digest.hexdigest()


def _environment_clearance_A(
    assembly: RNAFragmentAssembly,
    missing_residue_indices: frozenset[int],
    environment_atoms: tuple[dict[str, Any], ...],
    excluded_pairs: frozenset[tuple[int, str, str]],
    cutoff_A: float | None,
) -> tuple[float | None, bool]:
    if not environment_atoms:
        return None, False
    environment_xyz = np.asarray([atom["xyz_A"] for atom in environment_atoms], dtype=float)
    environment_ids = [atom["atom_id"] for atom in environment_atoms]
    minimum = None
    has_clash = False
    for atom in assembly.atoms:
        residue_index = atom["residue_index"]
        atom_name = atom["atom_name"]
        if residue_index not in missing_residue_indices:
            continue
        distances = np.linalg.norm(environment_xyz - np.asarray(atom["xyz_A"], dtype=float), axis=1)
        for index, environment_id in enumerate(environment_ids):
            if (residue_index, atom_name, environment_id) in excluded_pairs:
                continue
            distance = float(distances[index])
            minimum = distance if minimum is None else min(minimum, distance)
            if cutoff_A is not None and distance < cutoff_A:
                has_clash = True
    return minimum, has_clash


def _pin_gap_assembly(assembly, anchors, max_bond_change_A):
    original = {(a["residue_index"], a["atom_name"]): np.asarray(a["xyz_A"])
                for a in assembly.atoms}
    pinned = dict(original)
    for key in original:
        index, name = key
        if index in anchors:
            if name not in anchors[index]:
                raise ValueError("Missing observed atom in a pinned RNA anchor.")
            pinned[key] = anchors[index][name]
    for bond in assembly.bonds:
        a = (bond["residue_index_1"], bond["atom_1"])
        b = (bond["residue_index_2"], bond["atom_2"])
        if a[0] not in anchors and b[0] not in anchors:
            continue
        change = abs(np.linalg.norm(pinned[a]-pinned[b]) - np.linalg.norm(original[a]-original[b]))
        if change > max_bond_change_A:
            raise ValueError("Pinned RNA anchor would change an assembled bond beyond the limit.")
    return replace(assembly, atoms=tuple(
        dict(atom, xyz_A=pinned[(atom["residue_index"], atom["atom_name"])].tolist())
        for atom in assembly.atoms))


def _missing_geometry_key(assembly):
    """Fingerprint missing atoms on a fixed 0.001 Å grid without alignment."""
    rows = sorted((a["residue_index"], a["atom_name"],
                   tuple(int(v) for v in np.rint(np.asarray(a["xyz_A"]) * 1000)))
                  for a in assembly.atoms if 0 < a["residue_index"] < len(assembly.sequence)-1)
    return hashlib.sha256(json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("ascii")).digest()


def close_rna_gap_from_anchors(
    library: RNAFragmentLibrary,
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
    environment_atoms: Any = None,
    min_environment_distance_A: float | None = None,
    excluded_environment_pairs: Any = (),
    pin_anchor_coordinates: bool = False,
    deduplicate_missing_geometry: bool = False,
    anchor_angle_windows_deg: Mapping[str, tuple[float, float]] | None = None,
) -> RNAGapClosureResult:
    """Search observed RNA suite paths between two fixed anchor residues.

    ``sequence`` includes both anchor residues and at least one missing
    residue. All suites are internal phosphodiester states because both ends
    continue into the surrounding polymer. Anchor atom mappings use angstroms.
    Search is deterministic depth-first enumeration in suite-ID order and is
    explicitly bounded. A candidate satisfies software geometry limits only;
    optional environment screening is a caller-defined heavy-atom distance
    filter, not a force field or independent structural validation.
    With pin_anchor_coordinates enabled, every assembled anchor atom must
    have an observed coordinate. Returned anchor coordinates are exact;
    affected bonds must also pass the assembly bond-change tolerance
    relative to the unpinned assembly.
    Opt-in deduplication ranks one representative per missing atom inventory
    and coordinate fingerprint (0.001 Å grid) before applying the return cap.
    Optional inclusive angle windows in degrees screen the two final pinned
    interfaces. Missing or degenerate requested angles reject the candidate.
    """
    if not isinstance(library, RNAFragmentLibrary):
        raise TypeError("library must be an RNAFragmentLibrary.")
    if not isinstance(pin_anchor_coordinates, bool):
        raise ValueError("pin_anchor_coordinates must be a bool.")
    if not isinstance(deduplicate_missing_geometry, bool):
        raise ValueError("deduplicate_missing_geometry must be a bool.")
    angle_windows = None
    if anchor_angle_windows_deg is not None:
        if not pin_anchor_coordinates or not isinstance(anchor_angle_windows_deg, Mapping) or not anchor_angle_windows_deg:
            raise ValueError("anchor_angle_windows_deg requires pinned anchors and a nonempty mapping.")
        angle_windows = {}
        for name, bounds in anchor_angle_windows_deg.items():
            if name not in {"c3_o3_p_deg", "o3_p_o5_deg", "p_o5_c5_deg"}:
                raise ValueError(f"Unsupported pinned interface angle: {name!r}.")
            try:
                lo, hi = map(float, bounds)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} requires two bounds in degrees.") from exc
            if not 0 <= lo < hi <= 180:
                raise ValueError(f"{name} requires 0 <= lower < upper <= 180 degrees.")
            angle_windows[name] = (lo, hi)
    if not isinstance(sequence, str) or len(sequence) < 3 or any(base not in _BASES for base in sequence):
        raise ValueError("sequence must contain at least three standard RNA A/C/G/U bases, including anchors.")
    if coordinate_unit not in _UNIT_TO_ANGSTROM:
        raise ValueError("coordinate_unit must be 'angstrom' or 'nm'.")
    limits = {
        "max_translation_residual_A": max_translation_residual_A,
        "max_rotation_residual_degrees": max_rotation_residual_degrees,
        "max_shared_residue_rmsd_A": max_shared_residue_rmsd_A,
        "max_shared_atom_displacement_A": max_shared_atom_displacement_A,
        "max_bond_length_deviation_A": max_bond_length_deviation_A,
        "max_anchor_atom_displacement_A": max_anchor_atom_displacement_A,
        "max_anchor_connection_length_deviation_A": max_anchor_connection_length_deviation_A,
    }
    for name, value in limits.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite non-negative number.")
    for name, value in {
        "max_translation_residual_A": max_translation_residual_A,
        "max_rotation_residual_degrees": max_rotation_residual_degrees,
        "max_shared_residue_rmsd_A": max_shared_residue_rmsd_A,
    }.items():
        if value == 0:
            raise ValueError(f"{name} must be positive.")
    if isinstance(max_search_states, bool) or not isinstance(max_search_states, int) or max_search_states < 1:
        raise ValueError("max_search_states must be a positive integer.")
    if (isinstance(max_returned_candidates, bool)
            or not isinstance(max_returned_candidates, int)
            or max_returned_candidates < 1):
        raise ValueError("max_returned_candidates must be a positive integer.")

    start_atoms = _validate_anchor_atom_map(start_anchor_atoms, name="start_anchor_atoms")
    end_atoms = _validate_anchor_atom_map(end_anchor_atoms, name="end_anchor_atoms")
    if "O3'" not in start_atoms:
        raise ValueError("start_anchor_atoms must include the fixed upstream O3' connection atom.")
    if "P" not in end_atoms:
        raise ValueError("end_anchor_atoms must include the fixed downstream P connection atom.")
    validated_environment, excluded_pairs, environment_digest = _validate_environment_atoms(
        environment_atoms,
        min_environment_distance_A,
        excluded_environment_pairs,
        sequence,
    )
    if min_environment_distance_A is not None:
        min_environment_distance_A = float(min_environment_distance_A)
    start_A = _validated_frame(
        start_frame, scale_to_angstrom=_UNIT_TO_ANGSTROM[coordinate_unit], name="start_frame"
    )
    end_A = _validated_frame(
        end_frame, scale_to_angstrom=_UNIT_TO_ANGSTROM[coordinate_unit], name="end_frame"
    )
    suite_groups = [
        tuple(sorted(library.get_suites(sequence[index:index + 2]), key=lambda suite: suite["suite_id"]))
        for index in range(len(sequence) - 1)
    ]
    gap_atom_inventory = {
        (step + atom["residue_index"], atom["atom_name"])
        for step, group in enumerate(suite_groups)
        for suite in group
        for atom in suite["atoms"]
        if 1 <= step + atom["residue_index"] < len(sequence) - 1
    }
    invalid_exclusions = [
        (residue_index, atom_name, atom_id)
        for residue_index, atom_name, atom_id in excluded_pairs
        if (residue_index, atom_name) not in gap_atom_inventory
    ]
    if invalid_exclusions:
        raise ValueError(
            "excluded environment pairs reference atoms not present in the gap atom inventory: "
            f"{invalid_exclusions!r}."
        )

    # Stack entries retain only the current path; no candidate coordinates are
    # averaged or moved, and expansion order is stable across runs.
    stack: list[
        tuple[int, tuple[RNAFragmentPlacement, ...], tuple[float, ...], np.ndarray, float | None]
    ] = [
        (0, (), (), start_A, None)
    ]
    expanded_states = 0
    search_complete = True
    total_closures = 0
    best_candidates: list[RNAGapClosureCandidate] = []
    seen_geometries: set[bytes] = set()
    retained_geometries: dict[bytes, RNAGapClosureCandidate] = {}
    rejections = {
        "endpoint_frame": 0,
        "shared_residue": 0,
        "anchor_atom_inventory": 0,
        "anchor_displacement": 0,
        "anchor_connection_geometry": 0,
        "assembly": 0,
        "environment_clash": 0,
    }
    if pin_anchor_coordinates:
        rejections["pinned_anchor_geometry"] = 0
    if angle_windows is not None:
        from .structure_qc import measure_link_coordinates

        rejections["pinned_anchor_angles"] = 0

    while stack:
        step, placements, shared_rmsds, current_frame, start_displacement_so_far = stack.pop()
        for suite in reversed(suite_groups[step]):
            if expanded_states >= max_search_states:
                search_complete = False
                stack.clear()
                break
            expanded_states += 1
            is_last = step == len(suite_groups) - 1
            placement = library.place_suite(
                suite,
                current_frame,
                coordinate_unit="angstrom",
                downstream_frame=end_A if is_last else None,
            )

            if step == 0:
                start_displacement_current = _anchor_displacement(placement, 0, start_atoms)
                if start_displacement_current is None:
                    rejections["anchor_atom_inventory"] += 1
                    continue
                if start_displacement_current > max_anchor_atom_displacement_A:
                    rejections["anchor_displacement"] += 1
                    continue
            else:
                start_displacement_current = start_displacement_so_far

            if placements:
                shared = library._shared_residue_rmsd_A(placements[-1], placement)
                if shared is None or shared > max_shared_residue_rmsd_A:
                    rejections["shared_residue"] += 1
                    continue
            else:
                shared = None

            next_placements = placements + (placement,)
            next_shared = shared_rmsds + (() if shared is None else (float(shared),))
            if not is_last:
                stack.append((
                    step + 1, next_placements, next_shared, placement.downstream_frame,
                    start_displacement_current,
                ))
                continue

            assert placement.translation_residual_A is not None
            assert placement.rotation_residual_degrees is not None
            if (placement.translation_residual_A > max_translation_residual_A
                    or placement.rotation_residual_degrees > max_rotation_residual_degrees):
                rejections["endpoint_frame"] += 1
                continue
            end_displacement = _anchor_displacement(placement, 1, end_atoms)
            if end_displacement is None:
                rejections["anchor_atom_inventory"] += 1
                continue
            if end_displacement > max_anchor_atom_displacement_A:
                rejections["anchor_displacement"] += 1
                continue

            suite_by_id = {item["suite_id"]: item for item in library._suites}
            first_source = suite_by_id[next_placements[0].suite_id]
            last_source = suite_by_id[next_placements[-1].suite_id]
            start_reference_length = _source_o3p_bond_length(first_source)
            end_reference_length = _source_o3p_bond_length(last_source)
            first_missing_phosphate = next(
                np.asarray(atom["xyz_A"], dtype=float)
                for atom in next_placements[0].atoms
                if atom["residue_index"] == 1 and atom["atom_name"] == "P"
            )
            last_missing_o3 = next(
                np.asarray(atom["xyz_A"], dtype=float)
                for atom in next_placements[-1].atoms
                if atom["residue_index"] == 0 and atom["atom_name"] == "O3'"
            )
            if start_reference_length is None or end_reference_length is None:
                rejections["anchor_connection_geometry"] += 1
                continue
            start_connection_length = float(np.linalg.norm(
                start_atoms["O3'"] - first_missing_phosphate
            ))
            end_connection_length = float(np.linalg.norm(
                last_missing_o3 - end_atoms["P"]
            ))
            start_connection_deviation = abs(start_connection_length - start_reference_length)
            end_connection_deviation = abs(end_connection_length - end_reference_length)
            if (start_connection_deviation > max_anchor_connection_length_deviation_A
                    or end_connection_deviation > max_anchor_connection_length_deviation_A):
                rejections["anchor_connection_geometry"] += 1
                continue

            path = RNAFragmentPathSelection(
                sequence=sequence,
                placements=next_placements,
                shared_residue_rmsd_A=next_shared,
                normalized_score=0.0,
            )
            try:
                assembly = library.assemble_suite_path(
                    path,
                    max_shared_atom_displacement_A=max_shared_atom_displacement_A,
                    max_bond_length_deviation_A=max_bond_length_deviation_A,
                )
            except ValueError:
                rejections["assembly"] += 1
                continue

            if pin_anchor_coordinates:
                try:
                    assembly = _pin_gap_assembly(
                        assembly, {0: start_atoms, len(sequence)-1: end_atoms},
                        max_bond_length_deviation_A)
                except ValueError:
                    rejections["pinned_anchor_geometry"] += 1
                    continue
                coords = {(a["residue_index"], a["atom_name"]): np.asarray(a["xyz_A"])
                          for a in assembly.atoms}
                start_connection_length = float(np.linalg.norm(start_atoms["O3'"] - coords[(1, "P")]))
                end_connection_length = float(np.linalg.norm(
                    coords[(len(sequence)-2, "O3'")] - end_atoms["P"]))
                start_connection_deviation = abs(start_connection_length - start_reference_length)
                end_connection_deviation = abs(end_connection_length - end_reference_length)
                if max(start_connection_deviation, end_connection_deviation) > max_anchor_connection_length_deviation_A:
                    rejections["pinned_anchor_geometry"] += 1
                    continue
                if angle_windows is not None:
                    def residue_coords(index):
                        return {name: xyz for (i, name), xyz in coords.items() if i == index}

                    interfaces = (measure_link_coordinates(residue_coords(0), residue_coords(1)),
                                  measure_link_coordinates(residue_coords(len(sequence)-2),
                                                           residue_coords(len(sequence)-1)))
                    if any(link[name] is None or not lo <= link[name] <= hi for link in interfaces
                           for name, (lo, hi) in angle_windows.items()):
                        rejections["pinned_anchor_angles"] += 1
                        continue

            environment_min_distance, environment_clash = _environment_clearance_A(
                assembly,
                frozenset(range(1, len(sequence) - 1)),
                validated_environment,
                excluded_pairs,
                min_environment_distance_A,
            )
            if environment_clash:
                rejections["environment_clash"] += 1
                continue

            score_components = {
                "endpoint_translation": (
                    placement.translation_residual_A / max_translation_residual_A
                ) ** 2,
                "endpoint_rotation": (
                    placement.rotation_residual_degrees / max_rotation_residual_degrees
                ) ** 2,
                "shared_residue_continuity": sum(
                    (value / max_shared_residue_rmsd_A) ** 2 for value in next_shared
                ),
                "start_anchor_deviation": (
                    (start_displacement_current or 0.0)
                    / max(1e-12, max_anchor_atom_displacement_A or 1.0)
                ) ** 2,
                "end_anchor_deviation": (
                    end_displacement
                    / max(1e-12, max_anchor_atom_displacement_A or 1.0)
                ) ** 2,
                "start_anchor_connection_bond_deviation": (
                    start_connection_deviation
                    / max(1e-12, max_anchor_connection_length_deviation_A or 1.0)
                ) ** 2,
                "end_anchor_connection_bond_deviation": (
                    end_connection_deviation
                    / max(1e-12, max_anchor_connection_length_deviation_A or 1.0)
                ) ** 2,
                "assembled_shared_atom_deviation": (
                    assembly.shared_residue_max_displacement_A
                    / max(1e-12, max_shared_atom_displacement_A or 1.0)
                ) ** 2,
                "assembled_bond_length_deviation": (
                    assembly.max_bond_length_deviation_A
                    / max(1e-12, max_bond_length_deviation_A or 1.0)
                ) ** 2,
            }
            score = sum(score_components.values())
            candidate = RNAGapClosureCandidate(
                path_selection=RNAFragmentPathSelection(
                    sequence=sequence,
                    placements=next_placements,
                    shared_residue_rmsd_A=next_shared,
                    normalized_score=float(score),
                ),
                assembly=assembly,
                endpoint_translation_residual_A=placement.translation_residual_A,
                endpoint_rotation_residual_degrees=placement.rotation_residual_degrees,
                start_anchor_max_displacement_A=float(
                    start_displacement_current if start_displacement_current is not None else 0.0
                ),
                end_anchor_max_displacement_A=float(end_displacement),
                start_anchor_connection_reference_length_A=float(start_reference_length),
                end_anchor_connection_reference_length_A=float(end_reference_length),
                start_anchor_connection_length_A=start_connection_length,
                end_anchor_connection_length_A=end_connection_length,
                start_anchor_connection_length_deviation_A=float(start_connection_deviation),
                end_anchor_connection_length_deviation_A=float(end_connection_deviation),
                environment_min_distance_A=environment_min_distance,
                score_components=score_components,
                normalized_score=float(score),
            )
            total_closures += 1
            if deduplicate_missing_geometry:
                key = _missing_geometry_key(assembly)
                seen_geometries.add(key)
                previous = retained_geometries.get(key)
                if previous is not None:
                    if (candidate.normalized_score, candidate.suite_ids) >= (
                        previous.normalized_score, previous.suite_ids
                    ):
                        continue
                    best_candidates = [item for item in best_candidates if item is not previous]
                retained_geometries[key] = candidate
            best_candidates.append(candidate)
            best_candidates.sort(key=lambda item: (item.normalized_score, item.suite_ids))
            if len(best_candidates) > max_returned_candidates:
                evicted = best_candidates.pop()
                if deduplicate_missing_geometry:
                    del retained_geometries[_missing_geometry_key(evicted.assembly)]

    status = (
        "search_budget_exhausted" if not search_complete else
        "no_solution_found" if total_closures == 0 else
        "complete"
    )
    return RNAGapClosureResult(
        sequence=sequence,
        candidates=tuple(best_candidates),
        status=status,
        search_complete=search_complete,
        expanded_states=expanded_states,
        total_compatible_closures=total_closures,
        candidates_truncated=(len(seen_geometries) if deduplicate_missing_geometry
                              else total_closures) > len(best_candidates),
        start_frame_sha256=_frame_digest(start_A, coordinate_unit),
        end_frame_sha256=_frame_digest(end_A, coordinate_unit),
        start_anchor_atoms_sha256=_anchor_atoms_digest(start_atoms),
        end_anchor_atoms_sha256=_anchor_atoms_digest(end_atoms),
        source_manifest_sha256=library.source_manifest_sha256,
        chemistry_registry_sha256=library.chemistry_registry_sha256,
        environment_atoms_sha256=environment_digest,
        limits={**limits, "min_environment_distance_A": min_environment_distance_A,
                "max_search_states": max_search_states,
                "max_returned_candidates": max_returned_candidates},
        rejections=rejections,
        anchors_pinned=pin_anchor_coordinates,
        distinct_missing_geometries=len(seen_geometries) if deduplicate_missing_geometry else None,
        anchor_angle_windows_deg=angle_windows,
    )
