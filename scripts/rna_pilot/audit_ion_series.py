"""Audit same-sequence RNA ion-series coordinates, validation reports, and maps.

This script summarizes archived wwPDB validation evidence and verifies that the
RCSB map coefficients can be transformed into finite density grids. It does not
score a newly generated MDNA model against density or establish universal RNA
geometry.

Run in the project WSL ``dip2`` environment::

    python scripts/rna_pilot/audit_ion_series.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
import gemmi
import numpy as np


ENTRY_IDS = ("4u3l", "4u3o", "4u3p", "4u3r", "4u47", "4u78")
RNA_NAMES = {"A", "C", "G", "U"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cif_value(block: gemmi.cif.Block, tag: str) -> str | None:
    value = block.find_value(tag)
    if value in (None, ".", "?"):
        return None
    return gemmi.cif.as_string(value).strip()


def audit_coordinates(path: Path) -> dict[str, Any]:
    structure = gemmi.read_structure(str(path))
    if len(structure) != 1:
        raise ValueError(f"{path.name}: expected one model, found {len(structure)}")
    model = structure[0]
    chains = []
    for chain in model:
        residues = [residue for residue in chain if residue.name.strip().upper() in RNA_NAMES]
        if not residues:
            continue
        sequence = "".join(
            gemmi.find_tabulated_residue(residue.name).one_letter_code
            for residue in residues
        )
        missing_o2 = [
            f"{residue.name}:{residue.seqid}"
            for residue in residues
            if not any(atom.name.strip() == "O2'" for atom in residue)
        ]
        link_distances = []
        missing_links = []
        for upstream, downstream in zip(residues, residues[1:]):
            o3 = next((atom for atom in upstream if atom.name.strip() == "O3'"), None)
            phosphate = next((atom for atom in downstream if atom.name.strip() == "P"), None)
            if o3 is None or phosphate is None:
                missing_links.append(f"{upstream.seqid}->{downstream.seqid}")
                continue
            xyz_o3 = np.array([o3.pos.x, o3.pos.y, o3.pos.z])
            xyz_p = np.array([phosphate.pos.x, phosphate.pos.y, phosphate.pos.z])
            link_distances.append(float(np.linalg.norm(xyz_o3 - xyz_p)))
        chains.append(
            {
                "chain_id": chain.name,
                "residues": len(residues),
                "sequence": sequence,
                "missing_O2_prime": missing_o2,
                "O3_prime_to_next_P_A": link_distances,
                "missing_phosphodiester_links": missing_links,
                "O3_prime_to_next_P_range_A": [min(link_distances), max(link_distances)]
                if link_distances
                else None,
            }
        )
    block = gemmi.cif.read_file(str(path)).sole_block()
    return {
        "entry_id": _cif_value(block, "_entry.id"),
        "title": _cif_value(block, "_struct.title"),
        "method": _cif_value(block, "_exptl.method"),
        "resolution_A": _cif_value(block, "_refine.ls_d_res_high"),
        "space_group": structure.spacegroup_hm,
        "unit_cell_A": [structure.cell.a, structure.cell.b, structure.cell.c],
        "rna_chains": chains,
    }


def audit_report(path: Path) -> dict[str, Any]:
    with fitz.open(path) as document:
        page_count = len(document)
        text = "\n".join(page.get_text() for page in document)
    rna_start = text.find("5.3.3\nRNA")
    rna_end = text.find("5.4\n", rna_start)
    rna_section = text[rna_start:rna_end] if rna_start >= 0 and rna_end > rna_start else ""
    fit_start = text.find("6.1\nProtein, DNA and RNA chains")
    fit_end = text.find("6.2\n", fit_start)
    fit_section = text[fit_start:fit_end] if fit_start >= 0 and fit_end > fit_start else ""

    resolution_match = re.search(r"Resolution\s*:\s*\n?([0-9.]+)\s*Å", text)
    backbone_outlier_match = re.search(
        r"All\s+\((\d+)\)\s+RNA backbone outliers are listed below:\s*(.*?)(?:\nThere are no RNA pucker|\n5\.4\n)",
        rna_section,
        flags=re.DOTALL,
    )
    backbone_outlier_count = (
        int(backbone_outlier_match.group(1))
        if backbone_outlier_match
        else 0
        if "There are no RNA backbone outliers to report." in rna_section
        else None
    )
    clashscore_match = re.search(r"all-atom clashscore for this structure is\s+(\d+)", text)
    return {
        "report_file": path.name,
        "pages": page_count,
        "reported_resolution_A": float(resolution_match.group(1)) if resolution_match else None,
        "rna_section_found": bool(rna_section),
        "rna_backbone_no_outliers": "There are no RNA backbone outliers to report." in rna_section,
        "rna_backbone_outlier_count": backbone_outlier_count,
        "rna_backbone_outlier_details": backbone_outlier_match.group(2).strip()
        if backbone_outlier_match
        else None,
        "rna_pucker_no_outliers": "There are no RNA pucker outliers to report." in rna_section,
        "rna_validation_excerpt": rna_section.strip(),
        "no_chain_breaks": "There are no chain breaks in this entry." in text,
        "no_RSRZ_outliers": "There are no RSRZ outliers to report." in fit_section,
        "density_fit_excerpt": fit_section.strip(),
        "bond_length_no_outliers": "There are no bond length outliers." in text,
        "bond_angle_no_outliers": "There are no bond angle outliers." in text,
        "all_atom_clashscore": int(clashscore_match.group(1)) if clashscore_match else None,
        "clashscore_zero": bool(clashscore_match and int(clashscore_match.group(1)) == 0),
    }


def audit_coefficients(path: Path, f_tag: str, phi_tag: str) -> dict[str, Any]:
    block = gemmi.as_refln_blocks(gemmi.cif.read_file(str(path)))[0]
    labels = set(block.column_labels())
    if f_tag not in labels or phi_tag not in labels:
        raise ValueError(f"{path.name}: missing {f_tag}/{phi_tag}; has {sorted(labels)}")

    h = block.make_int_array("index_h", 0)
    amplitudes = block.make_float_array(f_tag, math.nan)
    phases = block.make_float_array(phi_tag, math.nan)
    finite = np.isfinite(amplitudes) & np.isfinite(phases)
    if not finite.any():
        raise ValueError(f"{path.name}: no finite amplitude/phase pairs")

    grid = block.transform_f_phi_to_map(f_tag, phi_tag, sample_rate=3.0)
    density = np.asarray(grid)
    if density.size == 0 or not np.isfinite(density).all():
        raise ValueError(f"{path.name}: generated map is empty or non-finite")
    return {
        "file": path.name,
        "reflection_rows": int(len(h)),
        "finite_amplitude_phase_rows": int(finite.sum()),
        "columns": sorted(labels),
        "grid_shape": list(density.shape),
        "grid_sample_rate": 3.0,
        "density_min": float(density.min()),
        "density_max": float(density.max()),
        "density_mean": float(density.mean()),
        "density_std": float(density.std()),
    }


def audit_entry(root: Path, entry_id: str) -> dict[str, Any]:
    coordinate_path = root / f"{entry_id}.cif"
    pdf_path = root / f"{entry_id}_full_validation.pdf"
    twofo_path = root / f"{entry_id}_2fo-fc_map_coef.cif.gz"
    fofc_path = root / f"{entry_id}_fo-fc_map_coef.cif.gz"
    for path in (coordinate_path, pdf_path, twofo_path, fofc_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "entry_id": entry_id.upper(),
        "coordinates": audit_coordinates(coordinate_path),
        "validation": audit_report(pdf_path),
        "two_fo_fc_map": audit_coefficients(twofo_path, "pdbx_FWT", "pdbx_PHWT"),
        "fo_fc_map": audit_coefficients(fofc_path, "pdbx_DELFWT", "pdbx_DELPHWT"),
    }


def audit_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    results = []
    for item in manifest["files"]:
        path = root / item["file"]
        exists = path.is_file()
        actual_hash = sha256(path) if exists else None
        results.append(
            {
                "file": item["file"],
                "exists": exists,
                "expected_sha256": item["sha256"],
                "actual_sha256": actual_hash,
                "hash_matches": actual_hash == item["sha256"],
            }
        )
    return {
        "manifest_file": manifest_path.name,
        "manifest_file_count": len(manifest["files"]),
        "all_manifest_files_match": all(item["hash_matches"] for item in results),
        "files": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[2] / "artifacts/rna_pilot/reference/ion_series"
    parser.add_argument("--input-dir", type=Path, default=default_root)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = args.input_dir.resolve()
    entries = [audit_entry(root, entry_id) for entry_id in ENTRY_IDS]
    sequences = {
        chain["sequence"]
        for entry in entries
        for chain in entry["coordinates"]["rna_chains"]
    }
    series_integrity = {
        "all_entries_have_two_8nt_rna_chains": all(
            len(entry["coordinates"]["rna_chains"]) == 2
            and all(chain["residues"] == 8 for chain in entry["coordinates"]["rna_chains"])
            for entry in entries
        ),
        "unique_chain_sequences": sorted(sequences),
        "all_chains_share_the_same_sequence": len(sequences) == 1,
        "all_rna_residues_have_O2_prime": all(
            not chain["missing_O2_prime"]
            for entry in entries
            for chain in entry["coordinates"]["rna_chains"]
        ),
    }
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(root),
        "interpretation_limit": (
            "Summarizes deposited validation and verifies coefficient-to-map conversion. "
            "Does not validate an MDNA-generated model against density or establish universal RNA geometry."
        ),
        "manifest_audit": audit_manifest(root),
        "series_integrity": series_integrity,
        "entries": entries,
    }
    output = args.output or (root / "audit.json")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Manifest hashes match: {result['manifest_audit']['all_manifest_files_match']}")
    print(f"Series integrity: {series_integrity}")
    for entry in result["entries"]:
        chains = entry["coordinates"]["rna_chains"]
        report = entry["validation"]
        print(
            f"{entry['entry_id']}: chains={len(chains)}, "
            f"RNA-backbone-outliers={report['rna_backbone_outlier_count']}, "
            f"RNA-pucker-outliers-free={report['rna_pucker_no_outliers']}, "
            f"chain-break-free={report['no_chain_breaks']}, "
            f"RSRZ-outliers-free={report['no_RSRZ_outliers']}, "
            f"clashscore={report['all_atom_clashscore']}, "
            f"O3-P-ranges={[chain['O3_prime_to_next_P_range_A'] for chain in chains]}, "
            f"map-grids={entry['two_fo_fc_map']['grid_shape']}/{entry['fo_fc_map']['grid_shape']}"
        )


if __name__ == "__main__":
    main()
