"""Audit the local 1BNA duplex and identify source-covered gap windows."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.gap_benchmarks.extract_1kx5_dna_suites import RESIDUE_TO_BASE

DEFAULT_INPUT = PROJECT_ROOT / "mdna/atomic/ddd/1bna.pdb"
DEFAULT_LIBRARY = PROJECT_ROOT / "artifacts/gap_benchmarks/1kx5_dna_suites.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/gap_benchmarks/1bna_dna_structure_audit.json"
BASES = frozenset("ACGT")
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}
EXPECTED_HBONDS = {
    "AT": (("N6", "O4"), ("N1", "N3")),
    "TA": (("O4", "N6"), ("N3", "N1")),
    "GC": (("O6", "N4"), ("N1", "N3"), ("N2", "O2")),
    "CG": (("N4", "O6"), ("N3", "N1"), ("O2", "N2")),
}
DINUCLEOTIDES = tuple("".join(pair) for pair in itertools.product("ACGT", repeat=2))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atom_coordinates(residue: Any) -> dict[str, np.ndarray]:
    selected = {}
    for atom in residue:
        previous = selected.get(atom.name)
        rank = (float(atom.occ), atom.altloc in ("\x00", " "), atom.altloc == "A")
        old_rank = ((float(previous.occ), previous.altloc in ("\x00", " "), previous.altloc == "A")
                    if previous is not None else None)
        if previous is None or rank > old_rank:
            selected[atom.name] = atom
    result = {}
    for name, atom in selected.items():
        point = np.asarray([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)
        if not np.isfinite(point).all():
            raise ValueError(f"{residue.name}:{residue.seqid} atom {name} has non-finite coordinates")
        result[name] = point
    return result


def _contact_distances(first: dict[str, np.ndarray], second: dict[str, np.ndarray], pair: str) -> list[float]:
    contacts = EXPECTED_HBONDS.get(pair)
    if contacts is None:
        return []
    distances = []
    for first_name, second_name in contacts:
        if first_name not in first or second_name not in second:
            raise ValueError(f"base pair {pair} lacks expected contact atoms {first_name}/{second_name}")
        distances.append(float(np.linalg.norm(first[first_name] - second[second_name])))
    return distances


def audit(input_path: Path = DEFAULT_INPUT, library_path: Path = DEFAULT_LIBRARY) -> dict[str, Any]:
    import gemmi

    structure = gemmi.read_structure(str(input_path))
    if len(structure) != 1:
        raise ValueError(f"Expected one coordinate model; found {len(structure)}")
    standard_by_chain = {}
    summaries = {}
    for chain in structure[0]:
        records = [(index, residue) for index, residue in enumerate(chain)
                   if residue.name in RESIDUE_TO_BASE]
        if not records:
            continue
        sequence = "".join(RESIDUE_TO_BASE[residue.name] for _, residue in records)
        if set(sequence) - BASES:
            raise ValueError(f"Chain {chain.name} has unsupported standard DNA identity")
        standard_by_chain[chain.name] = records
        link_lengths = []
        for (first_index, first), (second_index, second) in zip(records, records[1:]):
            if second_index != first_index + 1:
                continue
            first_atoms, second_atoms = _atom_coordinates(first), _atom_coordinates(second)
            if "O3'" in first_atoms and "P" in second_atoms:
                link_lengths.append(float(np.linalg.norm(first_atoms["O3'"] - second_atoms["P"])))
        summaries[chain.name] = {
            "standard_dna_residue_count": len(records),
            "sequence_5prime_to_3prime": sequence,
            "observed_adjacent_o3_prime_p_links": len(link_lengths),
            "o3_prime_p_distance_range_A": [min(link_lengths), max(link_lengths)] if link_lengths else None,
        }

    if not {"A", "B"} <= standard_by_chain.keys():
        raise ValueError("1BNA audit requires standard DNA chains A and B")
    chain_a, chain_b = standard_by_chain["A"], standard_by_chain["B"]
    sequence_a = summaries["A"]["sequence_5prime_to_3prime"]
    sequence_b = summaries["B"]["sequence_5prime_to_3prime"]
    if len(chain_a) != len(chain_b):
        raise ValueError("DNA chains A and B have different standard-residue counts")
    if sequence_a != "".join(COMPLEMENT[base] for base in reversed(sequence_b)):
        raise ValueError("Chains A and B do not match an antiparallel Watson-Crick sequence register")

    paired_rows = []
    for index, ((_, residue_a), (_, residue_b)) in enumerate(zip(chain_a, reversed(chain_b))):
        base_pair = sequence_a[index] + sequence_b[len(sequence_b) - 1 - index]
        coordinates_a = _atom_coordinates(residue_a)
        coordinates_b = _atom_coordinates(residue_b)
        distances = _contact_distances(coordinates_a, coordinates_b, base_pair)
        c1_a, c1_b = coordinates_a.get("C1'"), coordinates_b.get("C1'")
        if c1_a is None or c1_b is None:
            raise ValueError(f"Base pair {index} lacks a C1' atom")
        paired_rows.append({
            "chain_A_index_0based": index,
            "chain_B_index_0based": len(sequence_b) - 1 - index,
            "base_pair": base_pair,
            "expected_hbond_distances_A": distances,
            "c1_prime_separation_A": float(np.linalg.norm(c1_a - c1_b)),
        })

    library = json.loads(library_path.read_text(encoding="utf-8"))
    training_contexts = {
        context for context, count in library["suite_counts_by_sequence"].items() if count > 0
    }
    candidates = []
    n = len(chain_a)
    for missing_a in range(1, n - 1):
        missing_b = n - 1 - missing_a
        contexts_a = [sequence_a[missing_a - 1:missing_a + 1],
                      sequence_a[missing_a:missing_a + 2]]
        contexts_b = [sequence_b[missing_b - 1:missing_b + 1],
                      sequence_b[missing_b:missing_b + 2]]
        all_contexts = contexts_a + contexts_b
        if not all(context in training_contexts for context in all_contexts):
            continue
        flanking_indices = (missing_a - 1, missing_a + 1)
        flanking_pairs = [paired_rows[index] for index in flanking_indices]
        flanks_pass = all(
            row["expected_hbond_distances_A"]
            and max(row["expected_hbond_distances_A"]) <= 3.5
            for row in flanking_pairs
        )
        if not flanks_pass:
            continue
        candidates.append({
            "chain_A_missing_index_0based": missing_a,
            "chain_B_missing_index_0based": missing_b,
            "chain_A_triplet": sequence_a[missing_a - 1:missing_a + 2],
            "chain_B_triplet": sequence_b[missing_b - 1:missing_b + 2],
            "missing_base_pair": paired_rows[missing_a]["base_pair"],
            "training_contexts": {"chain_A": contexts_a, "chain_B": contexts_b},
            "four_contexts_present_in_1kx5": True,
            "flanking_expected_hbonds_within_3_5_A": True,
        })

    expected_distances = [distance for row in paired_rows
                          for distance in row["expected_hbond_distances_A"]]
    return {
        "schema_version": 1,
        "source_file": input_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "source_sha256": _sha256(input_path),
        "template_library_file": library_path.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "template_library_source_sha256": library["source_files"][0]["sha256"],
        "coordinate_unit": "angstrom",
        "chains": summaries,
        "pairing": {
            "registration": "chain_A[i] pairs with chain_B[n-1-i]",
            "sequence_matches_antiparallel_reverse_complement": True,
            "all_pairs_are_watson_crick": all(row["base_pair"] in EXPECTED_HBONDS for row in paired_rows),
            "pair_count": len(paired_rows),
            "expected_hbond_screen_cutoff_A": 3.5,
            "max_expected_hbond_distance_A": max(expected_distances),
            "max_c1_prime_separation_A": max(row["c1_prime_separation_A"] for row in paired_rows),
            "pairs": paired_rows,
        },
        "candidate_paired_gap_windows": candidates,
        "candidate_selection_policy": "Both strands must have both dinucleotide contexts in the 1KX5 library; immediate flanking base-pair expected contacts must all be within 3.5 A. This identifies candidates only; no coordinates were masked or solved.",
        "sources_represent_independent_families": False,
        "scientific_acceptance": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit(args.input, args.library)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "source_sha256": report["source_sha256"],
        "pair_count": report["pairing"]["pair_count"],
        "max_expected_hbond_distance_A": report["pairing"]["max_expected_hbond_distance_A"],
        "candidate_gap_windows": len(report["candidate_paired_gap_windows"]),
        "candidate_indices_A": [row["chain_A_missing_index_0based"]
                                for row in report["candidate_paired_gap_windows"]],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
