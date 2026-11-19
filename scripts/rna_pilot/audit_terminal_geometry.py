"""Compare observed RNA terminal geometry with same-base internal residues.

This audit quantifies whether the current internal-suite coordinates provide a
reasonable heavy-atom starting point for hydroxyl termini. It does not validate
terminal proton placement or prove that terminal coordinates can be transferred
between sequences or structures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import gemmi
import numpy as np


RNA_NAMES = {"A", "C", "G", "U"}
GEOMETRY = {
    "five_prime": {
        "reference_atoms": ("C4'", "C5'", "O5'"),
        "bond_atoms": ("C5'", "O5'"),
        "angle_atoms": ("C4'", "C5'", "O5'"),
        "terminal_substituent": "HO5'",
        "internal_substituent": "P",
    },
    "three_prime": {
        "reference_atoms": ("C4'", "C3'", "O3'"),
        "bond_atoms": ("C3'", "O3'"),
        "angle_atoms": ("C4'", "C3'", "O3'"),
        "terminal_substituent": "HO3'",
        "internal_substituent": "P",
    },
}


def _atom_map(residue: Any) -> dict[str, np.ndarray]:
    selected = {}
    for atom in residue:
        name = atom.name.strip()
        if atom.element.name.upper() in {"H", "D"}:
            selected.setdefault(name, atom)
            continue
        previous = selected.get(name)
        if previous is None or (atom.occ, atom.altloc in ("", " ", "\x00")) > (
            previous.occ, previous.altloc in ("", " ", "\x00")
        ):
            selected[name] = atom
    return {
        name: np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)
        for name, atom in selected.items()
    }


def _length(atoms: dict[str, np.ndarray], names: tuple[str, str]) -> float | None:
    if not all(name in atoms for name in names):
        return None
    return float(np.linalg.norm(atoms[names[0]] - atoms[names[1]]))


def _angle(atoms: dict[str, np.ndarray], names: tuple[str, str, str]) -> float | None:
    if not all(name in atoms for name in names):
        return None
    first, center, last = (atoms[name] for name in names)
    u, v = first - center, last - center
    denominator = np.linalg.norm(u) * np.linalg.norm(v)
    if denominator <= np.finfo(float).eps:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / denominator, -1.0, 1.0))))


def _terminal_record(chain_id: str, terminus: str, terminal: Any, internal: Any) -> dict[str, Any]:
    config = GEOMETRY[terminus]
    terminal_atoms = _atom_map(terminal)
    internal_atoms = _atom_map(internal)
    required = set(config["reference_atoms"])
    if not required.issubset(terminal_atoms) or not required.issubset(internal_atoms):
        raise ValueError(
            f"chain {chain_id} {terminus} comparison lacks atoms: "
            f"terminal={sorted(required - terminal_atoms.keys())}, "
            f"internal={sorted(required - internal_atoms.keys())}"
        )
    terminal_length = _length(terminal_atoms, config["bond_atoms"])
    internal_length = _length(internal_atoms, config["bond_atoms"])
    terminal_angle = _angle(terminal_atoms, config["angle_atoms"])
    internal_angle = _angle(internal_atoms, config["angle_atoms"])
    substituent = config["terminal_substituent"]
    return {
        "chain_id": chain_id,
        "terminus": terminus,
        "base": terminal.name.strip(),
        "terminal_residue_id": str(terminal.seqid),
        "internal_comparison_residue_id": str(internal.seqid),
        "terminal_substituent_observed": substituent in terminal_atoms,
        "internal_substituent_observed": config["internal_substituent"] in internal_atoms,
        "bond": {
            "atom_names": list(config["bond_atoms"]),
            "terminal_angstrom": terminal_length,
            "internal_angstrom": internal_length,
            "terminal_minus_internal_angstrom": terminal_length - internal_length,
        },
        "angle": {
            "atom_names": list(config["angle_atoms"]),
            "terminal_degrees": terminal_angle,
            "internal_degrees": internal_angle,
            "terminal_minus_internal_degrees": terminal_angle - internal_angle,
        },
    }


def audit(reference_dir: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(reference_dir.glob("*.cif")):
        raw = path.read_bytes()
        structure = gemmi.read_structure(str(path))
        if not structure:
            raise ValueError(f"{path} contains no coordinate models")
        chains = []
        for chain in structure[0]:
            residues = [residue for residue in chain if residue.name in RNA_NAMES]
            if len(residues) < 3:
                continue
            first, last = residues[0], residues[-1]
            first_internal = next(
                (residue for residue in residues[1:-1] if residue.name == first.name), None
            )
            last_internal = next(
                (residue for residue in reversed(residues[1:-1]) if residue.name == last.name), None
            )
            if first_internal is None or last_internal is None:
                continue
            chains.extend((
                _terminal_record(chain.name, "five_prime", first, first_internal),
                _terminal_record(chain.name, "three_prime", last, last_internal),
            ))
        if not chains:
            raise ValueError(f"{path} has no RNA chain with same-base internal comparisons")
        entries.append({
            "entry_id": path.stem.upper(),
            "source": f"https://files.rcsb.org/download/{path.stem.upper()}.cif",
            "file": path.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "chains": chains,
        })

    differences = {
        "five_prime_bond_angstrom": [],
        "five_prime_angle_degrees": [],
        "three_prime_bond_angstrom": [],
        "three_prime_angle_degrees": [],
    }
    for entry in entries:
        for item in entry["chains"]:
            prefix = item["terminus"]
            differences[f"{prefix}_bond_angstrom"].append(
                abs(item["bond"]["terminal_minus_internal_angstrom"])
            )
            differences[f"{prefix}_angle_degrees"].append(
                abs(item["angle"]["terminal_minus_internal_degrees"])
            )
    summary = {
        key: {
            "n": len(values),
            "median_absolute_difference": float(np.median(values)),
            "maximum_absolute_difference": float(np.max(values)),
        }
        for key, values in differences.items()
    }
    return {
        "method": (
            "For each RNA chain, compare the terminal residue with the nearest same-base "
            "internal residue using the listed local bond length and bond angle. No fitting "
            "or coordinates are modified."
        ),
        "entries": entries,
        "summary_absolute_differences": summary,
        "interpretation_limit": (
            "This six-entry same-sequence comparison describes selected local heavy-atom "
            "geometry only. It does not validate transfer to other sequences, terminal "
            "hydrogen placement, chemistry states, or a generated full chain."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("artifacts/rna_pilot/reference/ion_series"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/rna_pilot/reference/terminal_geometry_audit.json"),
    )
    args = parser.parse_args()
    report = audit(args.reference_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary_absolute_differences"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
