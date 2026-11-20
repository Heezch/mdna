"""Audit bundled RCSB CCD nucleotide records without modifying the inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import gemmi


SOURCE_TEMPLATE = "https://files.rcsb.org/ligands/download/{component_id}.cif"
PORT_ATOMS = ("P", "OP1", "OP2", "OP3", "O3'", "O5'", "HO3'", "HO5'")


def _cif_text(value: str) -> str:
    """Decode a CIF token, including quoted atom names such as ``O2'``."""

    return gemmi.cif.as_string(value).strip()


def audit_component(path: Path) -> dict[str, Any]:
    """Return an atom, bond, ideal-coordinate, and provenance summary."""

    raw = path.read_bytes()
    block = gemmi.cif.read_file(str(path)).sole_block()
    component_id = _cif_text(block.find_value("_chem_comp.id"))
    atom_rows = list(
        block.find(
            [
                "_chem_comp_atom.atom_id",
                "_chem_comp_atom.type_symbol",
                "_chem_comp_atom.charge",
                "_chem_comp_atom.pdbx_model_Cartn_x_ideal",
                "_chem_comp_atom.pdbx_model_Cartn_y_ideal",
                "_chem_comp_atom.pdbx_model_Cartn_z_ideal",
            ]
        )
    )
    bond_rows = list(
        block.find(
            [
                "_chem_comp_bond.atom_id_1",
                "_chem_comp_bond.atom_id_2",
                "_chem_comp_bond.value_order",
            ]
        )
    )

    atom_names = [_cif_text(row[0]) for row in atom_rows]
    atom_name_set = set(atom_names)
    missing_coordinates = []
    nonfinite_coordinates = []
    for atom_name, row in zip(atom_names, atom_rows):
        values = [row[index] for index in (3, 4, 5)]
        if any(value in (".", "?") for value in values):
            missing_coordinates.append(atom_name)
            continue
        if not all(math.isfinite(float(value)) for value in values):
            nonfinite_coordinates.append(atom_name)

    unknown_bond_atoms = sorted(
        {
            atom_name
            for row in bond_rows
            for atom_name in (_cif_text(row[0]), _cif_text(row[1]))
            if atom_name not in atom_name_set
        }
    )
    expected_id = path.stem.upper()

    return {
        "component_id": component_id,
        "expected_component_id": expected_id,
        "component_id_matches_filename": component_id == expected_id,
        "name": _cif_text(block.find_value("_chem_comp.name")),
        "component_type": _cif_text(block.find_value("_chem_comp.type")),
        "atom_count": len(atom_rows),
        "heavy_atom_count": sum(_cif_text(row[1]).upper() != "H" for row in atom_rows),
        "bond_count": len(bond_rows),
        "atom_names": atom_names,
        "has_ribose_O2_prime": "O2'" in atom_name_set,
        "port_atoms_present": [name for name in PORT_ATOMS if name in atom_name_set],
        "ideal_coordinates_complete": not missing_coordinates,
        "ideal_coordinates_finite": not nonfinite_coordinates,
        "missing_coordinate_atoms": missing_coordinates,
        "nonfinite_coordinate_atoms": nonfinite_coordinates,
        "unknown_bond_atoms": unknown_bond_atoms,
        "source_url": SOURCE_TEMPLATE.format(component_id=component_id),
        "source_file": path.name,
        "source_size_bytes": len(raw),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ccd-dir",
        type=Path,
        default=Path(__file__).parents[2]
        / "artifacts"
        / "rna_pilot"
        / "reference"
        / "ccd",
        help="Directory containing standard A.cif, C.cif, G.cif, and U.cif records.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. Without this option, print to stdout.",
    )
    args = parser.parse_args()

    records = [audit_component(args.ccd_dir / f"{code}.cif") for code in "ACGU"]
    report = {
        "audit_tool": "scripts/rna_pilot/audit_ccd_components.py",
        "gemmi_version": gemmi.__version__,
        "components": records,
        "all_component_ids_match": all(
            record["component_id_matches_filename"] for record in records
        ),
        "all_components_are_rna_linking": all(
            record["component_type"].upper() == "RNA LINKING" for record in records
        ),
        "all_ideal_coordinates_complete": all(
            record["ideal_coordinates_complete"] for record in records
        ),
        "all_bond_atoms_resolve": all(
            not record["unknown_bond_atoms"] for record in records
        ),
        "all_components_have_ribose_O2_prime": all(
            record["has_ribose_O2_prime"] for record in records
        ),
        "interpretation_limit": (
            "CCD component definitions describe chemical identity, internal bonds, "
            "and idealized single-component coordinates. They do not establish "
            "polymer conformation, path-building quality, force-field coverage, "
            "or experimental model quality."
        ),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote CCD audit: {args.output}")

    if not all(
        (
            report["all_component_ids_match"],
            report["all_components_are_rna_linking"],
            report["all_ideal_coordinates_complete"],
            report["all_bond_atoms_resolve"],
            report["all_components_have_ribose_O2_prime"],
        )
    ):
        raise SystemExit("CCD audit failed; inspect the JSON report for component-level details.")


if __name__ == "__main__":
    main()
