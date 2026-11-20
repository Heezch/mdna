"""Generate a reproducible ideal A-form dsRNA pilot with the local WinCoot API.

This is an adapter-level pilot, not an MDNA implementation. The result is an
idealized starting model and is not an experimental fit or scientific
acceptance result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "rna_pilot"
COOT_API_URL = (
    "https://www.mrc-lmb.cam.ac.uk/lucrezia/"
    "libcootapi-documentation/coot_api.html"
)
REFERENCE_ENTRY_URL = "https://www.rcsb.org/structure/4U3R"


def parse_pdb(path: Path) -> tuple[dict[str, list[dict[str, object]]], set[str]]:
    chains: dict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    atom_names: set[str] = set()
    with path.open("r", encoding="ascii", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            chain_id = line[21:22].strip() or "_"
            residue_number = int(line[22:26])
            residue_name = line[17:20].strip()
            xyz = tuple(float(line[i : i + 8]) for i in (30, 38, 46))
            atom_names.add(atom_name)
            residue = chains[chain_id].setdefault(
                residue_number,
                {"name": residue_name, "atoms": {}, "xyz": {}},
            )
            residue["atoms"][atom_name] = True
            residue["xyz"][atom_name] = xyz

    result: dict[str, list[dict[str, object]]] = {}
    for chain_id, residues in chains.items():
        result[chain_id] = [residues[key] for key in sorted(residues)]
    return result, atom_names


def summarize_geometry(
    chains: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    sequences: dict[str, str] = {}
    residues_per_chain: dict[str, int] = {}
    o3_p_distances: list[float] = []
    for chain_id, residues in chains.items():
        residues_per_chain[chain_id] = len(residues)
        sequences[chain_id] = "".join(str(residue["name"]) for residue in residues)
        for upstream, downstream in zip(residues, residues[1:]):
            upstream_xyz = upstream["xyz"]
            downstream_xyz = downstream["xyz"]
            if "O3'" not in upstream_xyz or "P" not in downstream_xyz:
                continue
            a = upstream_xyz["O3'"]
            b = downstream_xyz["P"]
            o3_p_distances.append(
                math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))
            )
    return {
        "chains": sorted(chains),
        "residues_per_chain": residues_per_chain,
        "residue_names_by_chain": sequences,
        "o3_prime_to_next_p_distance_angstrom": o3_p_distances,
        "o3_prime_to_next_p_distance_range_angstrom": (
            [min(o3_p_distances), max(o3_p_distances)] if o3_p_distances else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="UCGUACGA")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    sequence = args.sequence.upper()
    invalid = sorted(set(sequence) - set("ACGU"))
    if not sequence or invalid:
        parser.error(f"pilot sequence must contain only A/C/G/U; invalid: {invalid}")

    import coot  # Imported after the PowerShell launcher configures WinCoot paths.

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = (args.output_dir / f"ideal_A_dsRNA_{sequence}.pdb").resolve()
    manifest_path = pdb_path.with_suffix(".manifest.json")

    molecule_id = int(coot.ideal_nucleic_acid("RNA", "A", 0, sequence))
    if molecule_id < 0:
        raise RuntimeError("Coot ideal_nucleic_acid returned failure")

    writer_status = int(coot.write_pdb_file(molecule_id, str(pdb_path)))
    if not pdb_path.is_file() or pdb_path.stat().st_size == 0:
        raise RuntimeError(f"Coot did not write a non-empty PDB: {pdb_path}")

    chains, atom_names = parse_pdb(pdb_path)
    geometry = summarize_geometry(chains)
    missing = sorted({"O2'", "O3'", "P"} - atom_names)
    if missing:
        raise RuntimeError(f"RNA pilot is missing required RNA/backbone atoms: {missing}")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "idealized starting model; scientific validation pending",
        "generator": "WinCoot Python API ideal_nucleic_acid",
        "generator_api": COOT_API_URL,
        "python_executable": sys.executable,
        "coot_module": getattr(coot, "__file__", None),
        "coot_prefix": os.environ.get("COOT_PREFIX"),
        "coot_data_dir": os.environ.get("COOT_DATA_DIR"),
        "parameters": {
            "polymer": "RNA",
            "form": "A",
            "single_stranded_flag": 0,
            "input_sequence_5_to_3": sequence,
            "sequence_scope": "first strand; Coot generated the paired strand",
        },
        "coot_molecule_id": molecule_id,
        "coot_writer_status": writer_status,
        "coot_counts": {
            "chains": int(coot.n_chains(molecule_id)),
            "residues": int(coot.n_residues(molecule_id)),
            "atoms": int(coot.n_atoms(molecule_id)),
        },
        "pdb_summary": {
            **geometry,
            "atom_name_count": len(atom_names),
            "contains_ribose_o2_prime": "O2'" in atom_names,
        },
        "artifact": {
            "path": str(pdb_path.relative_to(PROJECT_ROOT)),
            "size_bytes": pdb_path.stat().st_size,
            "sha256": hashlib.sha256(pdb_path.read_bytes()).hexdigest(),
        },
        "reference_candidate": {
            "entry": "4U3R",
            "url": REFERENCE_ENTRY_URL,
            "role": "experimental comparison candidate; coordinates not locally audited",
        },
        "limitations": [
            "Ideal A-form initialization, not an experimental structure or fit.",
            "The generator covers standard RNA A/C/G/U only.",
            "No gap closure, modified nucleotides, XNAs, or force-field relaxation.",
            "The 4U3R reference coordinates have not yet been downloaded or compared.",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
