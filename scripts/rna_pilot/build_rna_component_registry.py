"""Build the offline RNA component data asset from audited RCSB CCD CIFs.

This development utility reads local source records only. It never accesses
the network and refuses to overwrite an existing asset unless --force is set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import gemmi


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CCD_DIR = ROOT / "artifacts" / "rna_pilot" / "reference" / "ccd"
DEFAULT_OUTPUT = ROOT / "mdna" / "data" / "rna_components.json"
SOURCE_TEMPLATE = "https://files.rcsb.org/ligands/download/{component_id}.cif"


def _value(raw: str) -> str:
    return gemmi.cif.as_string(raw).strip()


def _component_record(ccd_dir: Path, component_id: str, manifest: dict) -> dict:
    path = ccd_dir / f"{component_id}.cif"
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    expected = manifest["components"][component_id]["sha256"]
    if digest != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}: {digest} != {expected}")

    block = gemmi.cif.read_file(str(path)).sole_block()
    observed_id = _value(block.find_value("_chem_comp.id"))
    component_type = _value(block.find_value("_chem_comp.type"))
    if observed_id != component_id or component_type.upper() != "RNA LINKING":
        raise ValueError(f"Unexpected CCD identity/type in {path.name}: {observed_id}, {component_type}")

    atom_rows = block.find(
        [
            "_chem_comp_atom.atom_id",
            "_chem_comp_atom.type_symbol",
            "_chem_comp_atom.charge",
            "_chem_comp_atom.pdbx_model_Cartn_x_ideal",
            "_chem_comp_atom.pdbx_model_Cartn_y_ideal",
            "_chem_comp_atom.pdbx_model_Cartn_z_ideal",
        ]
    )
    atoms = []
    for row in atom_rows:
        atom_name = _value(row[0])
        coordinate_values = [row[index] for index in (3, 4, 5)]
        if any(value in (".", "?") for value in coordinate_values):
            raise ValueError(f"Missing ideal coordinate for {component_id}:{atom_name}")
        coordinate = [float(value) for value in coordinate_values]
        if not all(math.isfinite(value) for value in coordinate):
            raise ValueError(f"Non-finite ideal coordinate for {component_id}:{atom_name}")
        atoms.append(
            {
                "name": atom_name,
                "element": _value(row[1]),
                "charge": int(row[2]),
                "ideal_coordinate": coordinate,
            }
        )

    atom_names = {atom["name"] for atom in atoms}
    if "O2'" not in atom_names:
        raise ValueError(f"RNA component {component_id} is missing O2'")

    bond_rows = block.find(
        [
            "_chem_comp_bond.atom_id_1",
            "_chem_comp_bond.atom_id_2",
            "_chem_comp_bond.value_order",
        ]
    )
    bonds = []
    for row in bond_rows:
        atom_1, atom_2 = _value(row[0]), _value(row[1])
        if atom_1 not in atom_names or atom_2 not in atom_names:
            raise ValueError(f"Unknown bond endpoint in {component_id}: {atom_1}, {atom_2}")
        bonds.append({"atom_1": atom_1, "atom_2": atom_2, "order": _value(row[2]).lower()})

    return {
        "component_id": f"RNA:{component_id}",
        "polymer_type": "RNA",
        "sugar_chemistry": "ribose",
        "backbone_chemistry": "phosphodiester",
        "residue_name": component_id,
        "atoms": atoms,
        "bonds": bonds,
        "source": SOURCE_TEMPLATE.format(component_id=component_id),
        "sha256": digest,
        "coordinate_unit": "angstrom",
        "template_key": None,
    }


def build_registry(ccd_dir: Path) -> dict:
    manifest = json.loads((ccd_dir / "manifest.json").read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "source": manifest["source"],
        "source_documentation": manifest["source_documentation"],
        "retrieval_endpoint_template": manifest["retrieval_endpoint_template"],
        "retrieval_date": manifest["retrieval_date"],
        "coordinate_convention": "RCSB CCD ideal coordinates in angstrom",
        "components": [
            _component_record(ccd_dir, code, manifest) for code in "ACGU"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ccd-dir", type=Path, default=DEFAULT_CCD_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="Replace the existing generated asset.")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing asset: {args.output}; pass --force to replace it.")
    registry = build_registry(args.ccd_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(registry['components'])} components to {args.output}")


if __name__ == "__main__":
    main()
