import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
import mdtraj as md

from mdna.geometry import ReferenceBase
from scripts.gap_benchmarks.extract_1kx5_dna_suites import (
    make_base_frame,
    validate_suite,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIBRARY_PATH = PROJECT_ROOT / "artifacts/gap_benchmarks/1kx5_dna_suites.json"
SOURCE_PATH = PROJECT_ROOT / "docs/notebooks/pdbs/1kx5.pdb"
INDEPENDENT_SOURCE_PATH = PROJECT_ROOT / "mdna/atomic/ddd/1bna.pdb"
INDEPENDENT_LIBRARY_PATH = PROJECT_ROOT / "artifacts/gap_benchmarks/1bna_dna_suites.json"
ADDITIONAL_SOURCE_PATH = PROJECT_ROOT / "artifacts/gap_benchmarks/1wqy_source.pdb"
ADDITIONAL_LIBRARY_PATH = PROJECT_ROOT / "artifacts/gap_benchmarks/1wqy_dna_suites.json"
FOURTH_SOURCE_PATH = PROJECT_ROOT / "artifacts/gap_benchmarks/119d_source.pdb"
FOURTH_LIBRARY_PATH = PROJECT_ROOT / "artifacts/gap_benchmarks/119d_dna_suites.json"


def test_dna_base_frame_is_right_handed_and_requires_non_degenerate_anchors():
    coordinates = {
        "N1": np.array([0.0, 0.0, 0.0]),
        "C2": np.array([1.0, 0.0, 0.0]),
        "N3": np.array([1.5, 0.8, 0.0]),
        "C4": np.array([1.0, 1.6, 0.0]),
        "C5": np.array([0.0, 1.6, 0.0]),
        "C6": np.array([-0.5, 0.8, 0.0]),
        "C1'": np.array([-0.5, -0.5, 1.0]),
    }

    _, frame = make_base_frame("C", coordinates)

    assert frame.T @ frame == pytest.approx(np.eye(3), abs=1e-10)
    assert np.linalg.det(frame) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="lacks base-frame atoms"):
        make_base_frame("C", {"N1": np.zeros(3)})


@pytest.mark.parametrize("residue_name,base", [("DA", "A"), ("DC", "C"), ("DG", "G"), ("DT", "T")])
def test_dna_suite_frame_matches_mdna_reference_base_convention(residue_name, base):
    trajectory = md.load_pdb(str(PROJECT_ROOT / "docs/notebooks/pdbs/1kx5.pdb"))
    residue = next(residue for residue in trajectory.topology.residues if residue.name == residue_name)
    single = trajectory.atom_slice([atom.index for atom in residue.atoms])
    coordinates = {
        atom.name: trajectory.xyz[0, atom.index] * 10.0
        for atom in residue.atoms
    }

    origin, frame = make_base_frame(base, coordinates)
    reference = ReferenceBase(single)
    expected_frame = np.asarray(
        [reference.b_D[0], reference.b_L[0], reference.b_N[0]], dtype=float
    ).T

    assert origin == pytest.approx(reference.b_R[0] * 10.0, abs=5e-5)
    assert frame == pytest.approx(expected_frame, abs=2e-5)


def test_1kx5_dna_internal_suite_library_is_complete_and_source_traceable():
    report = json.loads(LIBRARY_PATH.read_text(encoding="utf-8"))

    assert report["suite_count"] == len(report["suites"]) == 288
    assert report["sequence_type_count"] == 15
    assert report["unobserved_sequence_dinucleotides"] == ["CG"]
    assert report["sources_represent_independent_families"] is False
    assert report["chemistry_registry"] == "mdna/chemistry.py:NUCLEOTIDE_REGISTRY"
    source_hash = hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest()
    assert report["source_files"] == [{
        "file": "docs/notebooks/pdbs/1kx5.pdb",
        "sha256": source_hash,
    }]
    assert all(suite["observed_O3_prime_to_next_P_A"] > 1.4 for suite in report["suites"])
    assert all((0, "O3'") in {(a["residue_index"], a["atom_name"]) for a in suite["atoms"]}
               and (1, "P") in {(a["residue_index"], a["atom_name"]) for a in suite["atoms"]}
               for suite in report["suites"])
    for suite in report["suites"]:
        validate_suite(suite)
        assert all(len(value) == 64 for value in suite["component_sha256"])


def test_dna_suite_extractor_preserves_identity_for_an_independent_local_structure():
    report = json.loads(INDEPENDENT_LIBRARY_PATH.read_text(encoding="utf-8"))

    assert report["suite_count"] > 0
    assert {suite["source_entry"] for suite in report["suites"]} == {"1BNA"}
    assert all(suite["suite_id"].startswith("1BNA:") for suite in report["suites"])
    assert report["source_files"][0]["file"] == "mdna/atomic/ddd/1bna.pdb"
    assert report["source_files"][0]["sha256"] == hashlib.sha256(
        INDEPENDENT_SOURCE_PATH.read_bytes()
    ).hexdigest()
    assert report["sources_represent_independent_families"] is False
    assert "1BNA" in report["interpretation_limit"]
    for suite in report["suites"]:
        validate_suite(suite)


def test_1wqy_standard_dna_suite_library_is_source_traceable_and_chemically_valid():
    report = json.loads(ADDITIONAL_LIBRARY_PATH.read_text(encoding="utf-8"))

    source_hash = hashlib.sha256(ADDITIONAL_SOURCE_PATH.read_bytes()).hexdigest()
    assert report["suite_count"] == len(report["suites"]) == 14
    assert report["sequence_type_count"] == 6
    assert report["unobserved_sequence_dinucleotides"] == [
        "AC", "AG", "CC", "CG", "CT", "GA", "GC", "GG", "GT", "TC"
    ]
    assert report["sources_represent_independent_families"] is False
    assert report["source_files"] == [{
        "file": "artifacts/gap_benchmarks/1wqy_source.pdb",
        "sha256": source_hash,
    }]
    assert {suite["source_entry"] for suite in report["suites"]} == {"1WQY"}
    assert all(suite["observed_O3_prime_to_next_P_A"] > 1.4 for suite in report["suites"])
    for suite in report["suites"]:
        validate_suite(suite)


def test_119d_standard_dna_suite_library_covers_ag_ct_and_is_source_traceable():
    report = json.loads(FOURTH_LIBRARY_PATH.read_text(encoding="utf-8"))

    source_hash = hashlib.sha256(FOURTH_SOURCE_PATH.read_bytes()).hexdigest()
    assert report["suite_count"] == len(report["suites"]) == 18
    assert report["sequence_type_count"] == 8
    assert report["suite_counts_by_sequence"]["AG"] == 2
    assert report["suite_counts_by_sequence"]["CT"] == 2
    assert report["sources_represent_independent_families"] is False
    assert report["source_files"] == [{
        "file": "artifacts/gap_benchmarks/119d_source.pdb",
        "sha256": source_hash,
    }]
    assert {suite["source_entry"] for suite in report["suites"]} == {"119D"}
    assert all(suite["observed_O3_prime_to_next_P_A"] > 1.4 for suite in report["suites"])
    for suite in report["suites"]:
        validate_suite(suite)
