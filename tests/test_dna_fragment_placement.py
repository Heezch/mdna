from pathlib import Path
import copy
import json

import mdtraj as md
import numpy as np
import pytest

from mdna import DNAFragmentAssembly, DNAFragmentLibrary, DNAFragmentPathSelection
from mdna.dna_fragments import UnsupportedDNASuiteError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIBRARY_PATH = PROJECT_ROOT / "artifacts/gap_benchmarks/1kx5_dna_suites.json"


def test_dna_fragment_library_places_source_suite_in_explicit_frame():
    library = DNAFragmentLibrary.from_json(LIBRARY_PATH)
    suite = library.get_suites("TC")[0]

    placement = library.place_suite(
        suite,
        np.array([[2.0, -1.0, 5.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        coordinate_unit="angstrom",
    )

    source_atom = next(atom for atom in suite["atoms"] if atom["residue_index"] == 0)
    placed_atom = next(atom for atom in placement.atoms if atom["residue_index"] == 0
                       and atom["atom_name"] == source_atom["atom_name"])
    assert np.asarray(placed_atom["xyz_A"]) == pytest.approx(
        np.asarray(source_atom["xyz_local_A"]) + np.array([2.0, -1.0, 5.0])
    )
    assert placement.downstream_frame.shape == (4, 3)
    assert placement.sequence == "TC"

    transform = suite["downstream_base_frame_relative_to_upstream"]
    target = np.vstack((
        transform["translation_A"],
        np.asarray(transform["rotation_matrix"], dtype=float).T,
    ))
    matched = library.place_suite(
        suite,
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        coordinate_unit="angstrom",
        downstream_frame=target,
    )
    assert matched.translation_residual_A == pytest.approx(0.0, abs=1e-10)
    assert matched.rotation_residual_degrees == pytest.approx(0.0, abs=1e-5)


def test_dna_fragment_library_rejects_unobserved_or_non_dna_sequences():
    library = DNAFragmentLibrary.from_json(LIBRARY_PATH)

    with pytest.raises(UnsupportedDNASuiteError, match="CG"):
        library.get_suites("CG")
    with pytest.raises(ValueError, match="A/C/G/T"):
        library.get_suites("AU")


def test_dna_fragment_library_rejects_atom_inventory_that_disagrees_with_component(tmp_path):
    data = json.loads(LIBRARY_PATH.read_text(encoding="utf-8"))
    data["suites"] = [copy.deepcopy(data["suites"][0])]
    data["suites"][0]["atoms"] = [
        atom for atom in data["suites"][0]["atoms"]
        if not (atom["residue_index"] == 0 and atom["atom_name"] == "C1'")
    ]
    source = tmp_path / "incomplete_dna_library.json"
    source.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="atom inventory"):
        DNAFragmentLibrary.from_json(source)


def test_dna_fragment_library_rejects_inconsistent_observed_link_distance(tmp_path):
    data = json.loads(LIBRARY_PATH.read_text(encoding="utf-8"))
    data["suites"] = [copy.deepcopy(data["suites"][0])]
    data["suites"][0]["observed_O3_prime_to_next_P_A"] += 0.1
    source = tmp_path / "inconsistent_link_distance.json"
    source.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="O3'-P distance does not match suite coordinates"):
        DNAFragmentLibrary.from_json(source)


def test_dna_fragment_library_selects_a_suite_matching_both_explicit_frames():
    library = DNAFragmentLibrary.from_json(LIBRARY_PATH)
    suite = library.get_suites("TC")[0]
    transform = suite["downstream_base_frame_relative_to_upstream"]
    frames = np.stack((
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        np.vstack((transform["translation_A"],
                   np.asarray(transform["rotation_matrix"], dtype=float).T)),
    ))

    path = library.select_suite_path(
        "TC", frames, coordinate_unit="angstrom",
        max_translation_residual_A=0.01,
        max_rotation_residual_degrees=0.1,
        max_shared_residue_rmsd_A=0.5,
    )

    assert isinstance(path, DNAFragmentPathSelection)
    assert path.suite_ids == (suite["suite_id"],)
    assert path.placements[0].translation_residual_A == pytest.approx(0.0, abs=1e-10)
    assert path.placements[0].rotation_residual_degrees == pytest.approx(0.0, abs=1e-5)
    report = path.to_report()
    assert report["algorithm"] == "dna-suite-dp-hard-limits-v1"
    assert report["source_files"] == list(library.source_files)
    assert report["input_base_frames_sha256"]
    assert report["limits"]["max_shared_residue_rmsd_A"] == 0.5
    assert report["placements"][0]["suite_id"] == suite["suite_id"]


def test_dna_fragment_library_joins_adjacent_source_suites_with_shared_residue_continuity(tmp_path):
    library = DNAFragmentLibrary.from_json(LIBRARY_PATH)
    suites = [suite for context in library.available_sequences
              for suite in library.get_suites(context)]
    first, second = next(
        (left, right) for left in suites for right in suites
        if left["chain_id"] == right["chain_id"]
        and left["source_residues"][1] == right["source_residues"][0]
    )
    sequence = first["sequence"] + second["sequence"][1]
    identity = np.eye(3)
    origins = [np.zeros(3)]
    bases = [identity]
    for suite in (first, second):
        transform = suite["downstream_base_frame_relative_to_upstream"]
        origins.append(origins[-1] + np.asarray(transform["translation_A"]) @ bases[-1])
        bases.append(np.asarray(transform["rotation_matrix"]).T @ bases[-1])
    frames = np.stack([np.vstack((origin, basis))
                       for origin, basis in zip(origins, bases)])

    path = library.select_suite_path(
        sequence, frames, coordinate_unit="angstrom",
        max_translation_residual_A=0.02,
        max_rotation_residual_degrees=0.2,
        max_shared_residue_rmsd_A=0.02,
    )

    assert path.suite_ids == (first["suite_id"], second["suite_id"])
    assert path.shared_residue_rmsd_A == pytest.approx((0.0,), abs=0.02)

    assembly = library.assemble_suite_path(
        path,
        max_shared_atom_displacement_A=0.02,
        max_bond_length_deviation_A=0.02,
    )
    assert isinstance(assembly, DNAFragmentAssembly)
    assert assembly.sequence == sequence
    assert assembly.five_prime_port == (0, "P")
    assert assembly.three_prime_port == (2, "O3'")
    assert len(assembly.shared_residue_rmsd_A) == 1
    assert {atom["residue_index"] for atom in assembly.atoms} == {0, 1, 2}
    inter_residue_bonds = [
        bond for bond in assembly.bonds
        if bond["residue_index_1"] != bond["residue_index_2"]
    ]
    assert len(inter_residue_bonds) == 2

    trajectory = assembly.to_mdtraj()
    expected_residue_names = [{"A": "DA", "C": "DC", "G": "DG", "T": "DT"}[base]
                              for base in sequence]
    assert [residue.name for residue in trajectory.topology.residues] == expected_residue_names
    assert trajectory.n_atoms == len(assembly.atoms)
    assert trajectory.topology.n_bonds == len(assembly.bonds)

    pdb_path = assembly.save_pdb(tmp_path / "assembled_tca.pdb")
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    cif = MMCIF2Dict(str(assembly.save_mmcif(tmp_path / "assembled_tca.cif")))
    assert cif["_entity_poly.type"] == ["polydeoxyribonucleotide"]
    assert cif["_entity_poly_seq.mon_id"] == expected_residue_names
    assert len(cif["_atom_site.id"]) == len(assembly.atoms)
    assert len(cif["_struct_conn.id"]) == len(assembly.bonds)
    links = [(a, b) for a, b, i, j in zip(
        cif["_struct_conn.ptnr1_label_atom_id"], cif["_struct_conn.ptnr2_label_atom_id"],
        cif["_struct_conn.ptnr1_label_seq_id"], cif["_struct_conn.ptnr2_label_seq_id"], strict=True)
        if i != j]
    assert links == [("O3'", "P"), ("O3'", "P")]
    restored = md.load_pdb(str(pdb_path))
    assert [residue.name for residue in restored.topology.residues] == expected_residue_names
    restored_links = [
        (bond[0].residue.index, bond[0].name, bond[1].residue.index, bond[1].name)
        for bond in restored.topology.bonds
        if bond[0].residue.index != bond[1].residue.index
    ]
    assert len(restored_links) == 2
    assert all(abs(first_residue - second_residue) == 1
               and {first_name, second_name} == {"O3'", "P"}
               for first_residue, first_name, second_residue, second_name in restored_links)

    displaced_path = copy.deepcopy(path)
    shared_atom = next(atom for atom in displaced_path.placements[1].atoms
                       if atom["residue_index"] == 0)
    shared_atom["xyz_A"][0] += 0.1
    with pytest.raises(ValueError, match="shared atom displacement"):
        library.assemble_suite_path(
            displaced_path,
            max_shared_atom_displacement_A=0.02,
            max_bond_length_deviation_A=0.02,
        )

    distorted_path = copy.deepcopy(path)
    slightly_displaced_atom = next(atom for atom in distorted_path.placements[1].atoms
                                   if atom["residue_index"] == 0 and atom["atom_name"] == "C1'")
    slightly_displaced_atom["xyz_A"][0] += 0.01
    with pytest.raises(ValueError, match="assembled bond length changed"):
        library.assemble_suite_path(
            distorted_path,
            max_shared_atom_displacement_A=0.02,
            max_bond_length_deviation_A=0.001,
        )
