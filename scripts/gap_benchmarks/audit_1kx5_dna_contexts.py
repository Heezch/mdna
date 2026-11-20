"""Audit standard DNA context coverage and backbone links in the local 1KX5 PDB."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

import gemmi
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "docs/notebooks/pdbs/1kx5.pdb"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/gap_benchmarks/1kx5_dna_contexts.json"
RESIDUE_TO_BASE = {"DA": "A", "DC": "C", "DG": "G", "DT": "T"}
DINUCLEOTIDES = tuple("".join(pair) for pair in itertools.product("ACGT", repeat=2))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atom_position(residue, atom_name: str) -> np.ndarray | None:
    matches = [atom for atom in residue if atom.name == atom_name]
    if not matches:
        return None
    # PDB alternate locations are not silently merged; select the highest
    # occupancy record and preserve the selected altloc in the report.
    atom = max(matches, key=lambda item: (item.occ, item.altloc == "A"))
    return np.asarray([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)


def audit(input_path: Path) -> dict:
    structure = gemmi.read_structure(str(input_path))
    if len(structure) == 0:
        raise ValueError(f"No coordinate model found in {input_path}.")

    model = structure[0]
    chains = []
    for chain in model:
        records = [
            (chain_index, residue)
            for chain_index, residue in enumerate(chain)
            if residue.name in RESIDUE_TO_BASE
        ]
        if not records:
            continue

        context_counts = Counter()
        sequence = "".join(RESIDUE_TO_BASE[residue.name] for _, residue in records)
        link_lengths = []
        missing_link_atoms = []
        nonconsecutive_chain_pairs = []
        for pair_index, ((first_chain_index, first), (second_chain_index, second)) in enumerate(
            zip(records, records[1:])
        ):
            context = RESIDUE_TO_BASE[first.name] + RESIDUE_TO_BASE[second.name]
            if second_chain_index != first_chain_index + 1:
                nonconsecutive_chain_pairs.append(pair_index)
                continue
            o3 = _atom_position(first, "O3'")
            phosphate = _atom_position(second, "P")
            if o3 is None or phosphate is None:
                missing_link_atoms.append({
                    "upstream": f"{first.seqid}:{first.name}",
                    "downstream": f"{second.seqid}:{second.name}",
                    "missing": [name for name, value in (("O3'", o3), ("P", phosphate)) if value is None],
                })
                continue
            context_counts[context] += 1
            link_lengths.append(float(np.linalg.norm(o3 - phosphate)))

        chains.append({
            "chain_id": chain.name,
            "standard_dna_residue_count": len(records),
            "sequence_5prime_to_3prime": sequence,
            "connected_dinucleotide_context_counts": {
                context: context_counts.get(context, 0) for context in DINUCLEOTIDES
            },
            "observed_connected_context_count": sum(context_counts.values()),
            "missing_link_atom_pairs": missing_link_atoms,
            "nonconsecutive_dna_residue_pairs_in_chain": nonconsecutive_chain_pairs,
            "o3_prime_p_distance_summary_A": (
                {
                    "count": len(link_lengths),
                    "minimum": min(link_lengths),
                    "mean": float(np.mean(link_lengths)),
                    "maximum": max(link_lengths),
                }
                if link_lengths else None
            ),
        })

    return {
        "schema_version": 1,
        "source_file": input_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "source_sha256": _sha256(input_path),
        "model_index_zero_based": 0,
        "coordinate_unit": "angstrom",
        "residue_filter": {"DA": "A", "DC": "C", "DG": "G", "DT": "T"},
        "link_evidence": "Observed adjacent chain residues with O3' and P coordinates; no geometric cutoff applied.",
        "chains": chains,
        "scientific_acceptance": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = audit(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "source_sha256": report["source_sha256"],
        "chains": [
            {
                "chain_id": chain["chain_id"],
                "residue_count": chain["standard_dna_residue_count"],
                "observed_context_count": chain["observed_connected_context_count"],
                "covered_context_types": sum(
                    value > 0 for value in chain["connected_dinucleotide_context_counts"].values()
                ),
            }
            for chain in report["chains"]
        ],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
