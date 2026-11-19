"""Extract source-traceable RNA dinucleotide suites containing one terminus.

The current six-structure pilot only supplies 5'-UC and 3'-GA boundary suites.
The output preserves terminal residue states and observed heavy-atom positions;
it does not imply coverage for other end sequences.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rna_pilot.extract_rna_suite_library import (  # noqa: E402
    ENTRY_IDS,
    sha256,
    extract_suite,
    validate_suite,
)


def extract_terminal_suite(
    entry_id: str,
    chain_id: str,
    upstream: Any,
    downstream: Any,
    source_hash: str,
    chain_sequence: str,
    terminal_role: str,
) -> dict[str, Any]:
    """Extract a boundary dinucleotide with its observed end chemistry."""
    if terminal_role not in {"five_prime", "three_prime"}:
        raise ValueError("terminal_role must be 'five_prime' or 'three_prime'.")
    return extract_suite(
        entry_id,
        chain_id,
        upstream,
        downstream,
        source_hash,
        chain_sequence,
        terminal_role=terminal_role,
    )


def build_terminal_library(reference_dir: Path) -> dict[str, Any]:
    import gemmi

    manifest_path = reference_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    manifest_hashes = {record["file"]: record["sha256"] for record in manifest["files"]}
    sources = []
    suites = []
    for entry_id in ENTRY_IDS:
        path = reference_dir / f"{entry_id}.cif"
        actual_hash = sha256(path)
        if manifest_hashes.get(path.name) != actual_hash:
            raise ValueError(f"{path.name}: SHA-256 does not match the reference manifest")
        structure = gemmi.read_structure(str(path))
        if len(structure) != 1:
            raise ValueError(f"{path.name}: expected one model, found {len(structure)}")
        sources.append({"file": path.name, "sha256": actual_hash})
        for chain in structure[0]:
            residues = [
                residue for residue in chain
                if residue.name.strip().upper() in "ACGU"
            ]
            if len(residues) < 3:
                continue
            sequence = "".join(residue.name.strip().upper() for residue in residues)
            suites.extend((
                extract_terminal_suite(
                    entry_id, chain.name, residues[0], residues[1], actual_hash,
                    sequence, "five_prime",
                ),
                extract_terminal_suite(
                    entry_id, chain.name, residues[-2], residues[-1], actual_hash,
                    sequence, "three_prime",
                ),
            ))

    for suite in suites:
        validate_suite(suite)
    counts = Counter(f"{suite['terminal_role']}:{suite['sequence']}" for suite in suites)
    link_distances = np.asarray(
        [suite["observed_O3_prime_to_next_P_A"] for suite in suites], dtype=float
    )
    return {
        "schema_version": 1,
        "library_type": "experimental terminal RNA dinucleotide suites",
        "coordinate_unit": "angstrom",
        "coordinate_policy": "observed heavy-atom coordinates; no hydrogen coordinates synthesized",
        "terminal_state_policy": (
            "5-prime suites use 5prime_hydroxyl on residue 0; 3-prime suites use "
            "3prime_hydroxyl on residue 1; the other residue is internally linked"
        ),
        "frame_definition": "upstream Tsukuba base frame; not a standard base-pair or X3DNA frame",
        "source_manifest": "artifacts/rna_pilot/reference/ion_series/manifest.json",
        "source_manifest_sha256": sha256(manifest_path),
        "chemistry_registry": "mdna/data/rna_components.json",
        "chemistry_registry_sha256": sha256(PROJECT_ROOT / "mdna/data/rna_components.json"),
        "sources": sources,
        "summary": {
            "suite_count": len(suites),
            "suite_counts_by_role_and_sequence": dict(sorted(counts.items())),
            "observed_O3_prime_to_next_P_A": {
                "count": int(len(link_distances)),
                "min": float(link_distances.min()),
                "max": float(link_distances.max()),
                "mean": float(link_distances.mean()),
            },
        },
        "suites": suites,
        "interpretation_limit": (
            "This library contains boundary suites from six crystal structures of one "
            "8-nt sequence. It supports only the observed boundary dinucleotides and "
            "does not establish cross-entry placement, general terminal chemistry, "
            "a complete RNA builder, or gap-closure performance."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts/rna_pilot/reference/ion_series",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/rna_pilot/reference/rna_terminal_suite_library.json",
    )
    args = parser.parse_args()
    report = build_terminal_library(args.reference_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
