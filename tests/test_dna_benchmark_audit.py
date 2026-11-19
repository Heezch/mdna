import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mdna.chemistry import get_nucleotide_component
from scripts.gap_benchmarks.evaluate_1bna_dna_gap_loo import _anchor_atoms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = PROJECT_ROOT / "artifacts/gap_benchmarks/1bna_dna_structure_audit.json"
SOURCE_PATH = PROJECT_ROOT / "mdna/atomic/ddd/1bna.pdb"


def test_1bna_audit_records_antiparallel_pairing_and_covered_gap_windows():
    report = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))

    assert report["source_sha256"] == hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()
    assert report["chains"]["A"]["sequence_5prime_to_3prime"] == "CGCGAATTCGCG"
    assert report["chains"]["B"]["sequence_5prime_to_3prime"] == "CGCGAATTCGCG"
    assert report["pairing"]["registration"] == "chain_A[i] pairs with chain_B[n-1-i]"
    assert report["pairing"]["all_pairs_are_watson_crick"] is True
    assert report["pairing"]["max_expected_hbond_distance_A"] < 3.5

    candidate = next(
        item for item in report["candidate_paired_gap_windows"]
        if item["chain_A_missing_index_0based"] == 4
    )
    assert candidate["chain_A_triplet"] == "GAA"
    assert candidate["chain_B_missing_index_0based"] == 7
    assert candidate["chain_B_triplet"] == "TTC"
    assert candidate["training_contexts"] == {
        "chain_A": ["GA", "AA"],
        "chain_B": ["TT", "TC"],
    }
    assert report["scientific_acceptance"] is False


def test_gap_anchor_extraction_rejects_incomplete_heavy_atom_inventory():
    component = get_nucleotide_component("A")
    atom_names = [atom.name for atom in component.atoms if atom.element.upper() != "H"]
    atom_names.remove("C1'")
    class Residue:
        name = "DA"
        seqid = "1"

        def __iter__(self):
            return iter([
                SimpleNamespace(
                    name=name,
                    occ=1.0,
                    altloc="\x00",
                    pos=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                )
                for name in atom_names
            ])

    with pytest.raises(ValueError, match="missing registered heavy atoms.*C1'"):
        _anchor_atoms(Residue())
