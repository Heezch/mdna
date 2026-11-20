"""Audit the RCSB 4U3R RNA reference against the local ideal A-form pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import gemmi
import numpy as np


RNA_NAMES = {"A", "C", "G", "U"}
BACKBONE_ATOMS = {
    "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'",
    "O3'", "C2'", "O2'", "C1'",
}


def _clean_cif_value(value: str) -> str:
    return value.strip().strip("'\"")


def _heavy_atom_map(chain):
    """Map residue/name pairs to the highest-occupancy heavy-atom position."""
    atoms = {}
    duplicate_keys = set()
    for residue in chain:
        for atom in residue:
            if atom.element.name.upper() in {"H", "D"}:
                continue
            key = (residue.seqid.num, atom.name)
            if key in atoms:
                duplicate_keys.add(key)
                old_atom = atoms[key][0]
                old_blank = old_atom.altloc in ("", " ", "\x00")
                new_blank = atom.altloc in ("", " ", "\x00")
                if (atom.occ, new_blank) <= (old_atom.occ, old_blank):
                    continue
            atoms[key] = (atom, np.array([atom.pos.x, atom.pos.y, atom.pos.z]))
    return {key: value[1] for key, value in atoms.items()}, duplicate_keys


def _residue_atom_inventory(residue):
    atom_names = sorted({atom.name for atom in residue})
    name_set = set(atom_names)
    return {
        "component_id": residue.name,
        "atom_names": atom_names,
        "has_P": "P" in name_set,
        "has_OP3": "OP3" in name_set,
        "has_HOP3": "HOP3" in name_set,
        "has_O3_prime": "O3'" in name_set,
        "has_HO3_prime": "HO3'" in name_set,
        "has_O5_prime": "O5'" in name_set,
        "has_HO5_prime": "HO5'" in name_set,
    }


def _align_ideal_chain(reference_chain, ideal_chain):
    reference_atoms, reference_alternates = _heavy_atom_map(reference_chain)
    ideal_atoms, ideal_alternates = _heavy_atom_map(ideal_chain)
    common = sorted(set(reference_atoms) & set(ideal_atoms))
    fit_atoms = [key for key in common if key[1] in BACKBONE_ATOMS]
    if not fit_atoms:
        raise ValueError("No shared backbone atoms are available for superposition.")

    target = np.array([reference_atoms[key] for key in fit_atoms])
    moving = np.array([ideal_atoms[key] for key in fit_atoms])
    target_center = target.mean(axis=0)
    moving_center = moving.mean(axis=0)
    u, _, vt = np.linalg.svd((moving - moving_center).T @ (target - target_center))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = target_center - moving_center @ rotation

    fit_coordinates = np.array([ideal_atoms[key] @ rotation + translation for key in fit_atoms])
    common_ideal = np.array([ideal_atoms[key] @ rotation + translation for key in common])
    common_reference = np.array([reference_atoms[key] for key in common])
    fit_rmsd = np.sqrt(np.mean(np.sum((fit_coordinates - target) ** 2, axis=1)))
    all_atom_rmsd = np.sqrt(
        np.mean(np.sum((common_ideal - common_reference) ** 2, axis=1))
    )
    return {
        "reference_chain": reference_chain.name,
        "ideal_chain": ideal_chain.name,
        "common_heavy_atom_count": len(common),
        "backbone_fit_atom_count": len(fit_atoms),
        "backbone_fit_rmsd_angstrom": float(fit_rmsd),
        "all_common_heavy_atom_rmsd_after_backbone_fit_angstrom": float(all_atom_rmsd),
        "reference_altloc_atom_name_pairs": len(reference_alternates),
        "ideal_altloc_atom_name_pairs": len(ideal_alternates),
        "fit_atom_names": sorted({key[1] for key in fit_atoms}),
    }


def audit(cif_path: Path, ideal_pdb_path: Path) -> dict:
    cif_bytes = cif_path.read_bytes()
    block = gemmi.cif.read_file(str(cif_path)).sole_block()
    structure = gemmi.make_structure_from_block(block)
    if not structure:
        raise ValueError("The mmCIF contains no coordinate models.")

    model = structure[0]
    rna_chains = []
    for chain in model:
        residues = [residue for residue in chain if residue.name in RNA_NAMES]
        if not residues:
            continue
        o3_to_next_p = []
        for first, second in zip(residues, residues[1:]):
            o3 = first.find_atom("O3'", "*")
            phosphate = second.find_atom("P", "*")
            o3_to_next_p.append(
                float(o3.pos.dist(phosphate.pos)) if o3 and phosphate else None
            )
        rna_chains.append(
            {
                "chain_id": chain.name,
                "sequence": "".join(residue.name for residue in residues),
                "residue_count": len(residues),
                "residue_names": [residue.name for residue in residues],
                "heavy_atom_count": sum(
                    1
                    for residue in residues
                    for atom in residue
                    if atom.element.name.upper() not in {"H", "D"}
                ),
                "all_residues_have_O2_prime": all(
                    residue.find_atom("O2'", "*") is not None for residue in residues
                ),
                "adjacent_O3_prime_to_P_distances_angstrom": o3_to_next_p,
                "five_prime_terminal_observed_atoms": _residue_atom_inventory(residues[0]),
                "three_prime_terminal_observed_atoms": _residue_atom_inventory(residues[-1]),
                "internal_residue_linkage_flags": [
                    _residue_atom_inventory(residue) for residue in residues[1:-1]
                ],
            }
        )

    ideal = gemmi.read_structure(str(ideal_pdb_path))
    reference_chain = next(
        (chain for chain in model if chain.name == "A" and any(r.name in RNA_NAMES for r in chain)),
        None,
    )
    ideal_chain = next(
        (chain for chain in ideal[0] if chain.name == "A" and any(r.name in RNA_NAMES for r in chain)),
        None,
    )
    if reference_chain is None or ideal_chain is None:
        raise ValueError("Chain A must be present in both the reference and ideal model.")
    if "".join(r.name for r in reference_chain if r.name in RNA_NAMES) != "".join(
        r.name for r in ideal_chain if r.name in RNA_NAMES
    ):
        raise ValueError("Chain A sequences differ; atomwise comparison is not valid.")

    non_rna_residues = Counter(
        residue.name
        for chain in model
        for residue in chain
        if residue.name not in RNA_NAMES
    )
    heavy_atom_count = sum(
        1
        for chain in model
        for residue in chain
        for atom in residue
        if atom.element.name.upper() not in {"H", "D"}
    )

    return {
        "audit_time_utc": datetime.now(timezone.utc).isoformat(),
        "entry_id": _clean_cif_value(block.find_value("_entry.id")),
        "title": _clean_cif_value(block.find_value("_struct.title")),
        "experimental_method": _clean_cif_value(block.find_value("_exptl.method")),
        "resolution_angstrom": float(_clean_cif_value(block.find_value("_refine.ls_d_res_high"))),
        "source": "https://files.rcsb.org/download/4u3r.cif",
        "local_cif": cif_path.name,
        "cif_size_bytes": len(cif_bytes),
        "cif_sha256": hashlib.sha256(cif_bytes).hexdigest(),
        "gemmi_version": gemmi.__version__,
        "coordinate_unit": "angstrom",
        "model_count": len(structure),
        "model_0_total_atom_count": sum(
            1 for chain in model for residue in chain for _ in residue
        ),
        "model_0_heavy_atom_count": heavy_atom_count,
        "rna_chains": rna_chains,
        "polymer_linkage_observations": {
            "observed_model_OP3_count": sum(
                1
                for chain in model
                for residue in chain
                if residue.name in RNA_NAMES
                for atom in residue
                if atom.name == "OP3"
            ),
            "observed_RNA_HO3_prime_count": sum(
                1
                for chain in model
                for residue in chain
                if residue.name in RNA_NAMES
                for atom in residue
                if atom.name == "HO3'"
            ),
            "observed_RNA_HO5_prime_count": sum(
                1
                for chain in model
                for residue in chain
                if residue.name in RNA_NAMES
                for atom in residue
                if atom.name == "HO5'"
            ),
            "interpretation": (
                "In this 4U3R model, each strand begins with an observed 5-prime OH state and "
                "ends with a 3-prime OH state; internal links are represented by adjacent "
                "O3-prime/P coordinates. The linking CCD components include OP3 and terminal "
                "hydrogens that are absent from linked internal residues. Terminal state and "
                "link atom handling must therefore be explicit in the builder."
            ),
        },
        "non_rna_residue_name_counts": dict(sorted(non_rna_residues.items())),
        "struct_conn_row_count": len(block.find_mmcif_category("_struct_conn.")),
        "chain_A_comparison_to_ideal_A_form": _align_ideal_chain(
            reference_chain, ideal_chain
        ),
        "comparison_method": (
            "Least-squares rigid superposition of shared chain-A sugar/backbone heavy atoms; "
            "RMSD then computed over shared heavy atoms."
        ),
        "interpretation_limit": (
            "Descriptive coordinate comparison only; not an energy, force-field, "
            "density-fit, or general RNA-quality validation."
        ),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cif",
        type=Path,
        default=project_root / "artifacts/rna_pilot/reference/4U3R.cif",
    )
    parser.add_argument(
        "--ideal-pdb",
        type=Path,
        default=project_root / "artifacts/rna_pilot/ideal_A_dsRNA_UCGUACGA.pdb",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "artifacts/rna_pilot/reference/4U3R.audit.json",
    )
    args = parser.parse_args()
    result = audit(args.cif, args.ideal_pdb)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
