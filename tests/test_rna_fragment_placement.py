import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import mdtraj as md

from mdna import build_rna_chain_from_frames
from mdna.rna_fragments import (
    NoCompatibleRNAFragmentPathError,
    RNAFragmentAssembly,
    RNAFragmentLibrary,
    RNAFragmentPathSelection,
    UnsupportedSuiteSequenceError,
)


def _suite_record(sequence="AC"):
    identity = np.eye(3)
    return {
        "suite_id": "synthetic:A:1-2",
        "sequence": sequence,
        "coordinate_unit": "angstrom",
        "residue_states": [
            "5prime_phosphodiester+3prime_phosphodiester",
            "5prime_phosphodiester+3prime_phosphodiester",
        ],
        "atoms": [
            {
                "residue_index": 0,
                "atom_name": "O3'",
                "element": "O",
                "xyz_local_A": [0.0, 0.0, 0.0],
            },
            {
                "residue_index": 1,
                "atom_name": "P",
                "element": "P",
                "xyz_local_A": [1.6, 2.0, 3.0],
            },
        ],
        "bonds": [
            {
                "residue_index_1": 0,
                "atom_1": "O3'",
                "residue_index_2": 1,
                "atom_2": "P",
                "order": "single",
            }
        ],
        "downstream_base_frame_relative_to_upstream": {
            "translation_A": [3.0, -2.0, 4.0],
            "rotation_matrix": identity.tolist(),
        },
    }


def _identity_frame(origin=(0.0, 0.0, 0.0)):
    return np.vstack((np.asarray(origin, dtype=float), np.eye(3)))


def _path_suite(sequence, suite_id, *, shared_residue_offset=0.0,
                upstream_residue_offset=0.0, frame_step=3.0):
    atoms = []
    names = (("C1'", "C", [0.0, 0.0, 0.0]), ("N1", "N", [0.0, 1.0, 0.0]),
             ("C2", "C", [0.0, 0.0, 1.0]), ("O3'", "O", [0.0, 0.0, 2.0]))
    for atom_name, element, xyz in names:
        upstream_xyz = np.asarray(xyz, dtype=float).copy()
        if atom_name == "C1'":
            upstream_xyz[0] += upstream_residue_offset
        atoms.append({"residue_index": 0, "atom_name": atom_name, "element": element,
                      "xyz_local_A": upstream_xyz.tolist()})
        downstream_xyz = np.asarray(xyz, dtype=float) + np.array([frame_step, 0.0, 0.0])
        if atom_name == "C1'":
            downstream_xyz[0] += shared_residue_offset
        atoms.append({"residue_index": 1, "atom_name": atom_name, "element": element,
                      "xyz_local_A": downstream_xyz.tolist()})
    atoms.append({"residue_index": 0, "atom_name": "P", "element": "P",
                  "xyz_local_A": [0.0, 0.0, 3.0]})
    atoms.append({"residue_index": 1, "atom_name": "P", "element": "P",
                  "xyz_local_A": [frame_step, 0.0, 3.0]})
    return {
        "suite_id": suite_id,
        "sequence": sequence,
        "coordinate_unit": "angstrom",
        "residue_states": [
            "5prime_phosphodiester+3prime_phosphodiester",
            "5prime_phosphodiester+3prime_phosphodiester",
        ],
        "atoms": atoms,
        "bonds": [{"residue_index_1": 0, "atom_1": "O3'", "residue_index_2": 1,
                   "atom_2": "P", "order": "single"}],
        "downstream_base_frame_relative_to_upstream": {
            "translation_A": [frame_step, 0.0, 0.0],
            "rotation_matrix": np.eye(3).tolist(),
        },
    }


def test_library_loads_provenance_artifact_and_reports_observed_coverage():
    project_root = Path(__file__).resolve().parents[1]
    artifact = project_root / "artifacts/rna_pilot/reference/rna_suite_library.json"

    library = RNAFragmentLibrary.from_json(artifact)

    assert library.available_sequences == ("AC", "CG", "GU", "UA")
    assert len(library.get_suites("CG")) == 24
    assert library.source_manifest_sha256
    assert library.chemistry_registry_sha256


def test_library_rejects_missing_and_noncanonical_suite_sequences():
    library = RNAFragmentLibrary([_suite_record()])

    with pytest.raises(UnsupportedSuiteSequenceError, match="AC"):
        library.get_suites("AA")
    with pytest.raises(ValueError, match="standard RNA A/C/G/U"):
        library.get_suites("AX")


def test_returned_suite_records_cannot_mutate_the_library():
    library = RNAFragmentLibrary([_suite_record()])
    suite = library.get_suites("AC")[0]
    suite["atoms"][0]["xyz_local_A"][0] = 99.0

    assert library.get_suites("AC")[0]["atoms"][0]["xyz_local_A"][0] == 0.0


def test_library_rejects_terminal_chemistry_in_internal_suite_collection():
    record = _suite_record()
    record["residue_states"][0] = "5prime_hydroxyl+3prime_phosphodiester"

    with pytest.raises(ValueError, match="terminal role and residue states disagree"):
        RNAFragmentLibrary([record])


@pytest.mark.parametrize(
    "role,states",
    [
        ("five_prime", [
            "5prime_hydroxyl+3prime_phosphodiester",
            "5prime_phosphodiester+3prime_phosphodiester",
        ]),
        ("three_prime", [
            "5prime_phosphodiester+3prime_phosphodiester",
            "5prime_phosphodiester+3prime_hydroxyl",
        ]),
    ],
)
def test_library_accepts_explicit_terminal_suite_states(role, states):
    record = _suite_record("UC" if role == "five_prime" else "GA")
    record["terminal_role"] = role
    record["residue_states"] = states
    terminal_atom = "O5'" if role == "five_prime" else "O3'"
    terminal_residue = 0 if role == "five_prime" else 1
    record["atoms"].append({
        "residue_index": terminal_residue,
        "atom_name": terminal_atom,
        "element": "O",
        "xyz_local_A": [0.0, 0.0, 0.0],
    })

    library = RNAFragmentLibrary([record])

    assert library.get_suites(record["sequence"], terminal_role=role)[0]["terminal_role"] == role
    with pytest.raises(UnsupportedSuiteSequenceError):
        library.get_suites(record["sequence"])


def test_terminal_suite_path_assembles_hydroxyl_ends_without_open_ports():
    five_prime = _path_suite("UC", "UC:5prime")
    five_prime["terminal_role"] = "five_prime"
    five_prime["residue_states"] = [
        "5prime_hydroxyl+3prime_phosphodiester",
        "5prime_phosphodiester+3prime_phosphodiester",
    ]
    five_prime["atoms"] = [
        atom for atom in five_prime["atoms"]
        if not (atom["residue_index"] == 0 and atom["atom_name"] in {"P", "OP1", "OP2"})
    ]
    three_prime = _path_suite("CG", "CG:3prime")
    three_prime["terminal_role"] = "three_prime"
    three_prime["residue_states"] = [
        "5prime_phosphodiester+3prime_phosphodiester",
        "5prime_phosphodiester+3prime_hydroxyl",
    ]
    for record in (five_prime, three_prime):
        for residue_index in (0, 1):
            for atom_name, element, xyz in (
                ("O2'", "O", [3.0 * residue_index + 0.5, 0.5, 0.0]),
                ("O5'", "O", [3.0 * residue_index + 0.5, 0.0, 0.5]),
            ):
                if not any(atom["residue_index"] == residue_index and atom["atom_name"] == atom_name
                           for atom in record["atoms"]):
                    record["atoms"].append({
                        "residue_index": residue_index, "atom_name": atom_name,
                        "element": element, "xyz_local_A": xyz,
                    })
    library = RNAFragmentLibrary([five_prime, three_prime])
    frames = np.stack([_identity_frame((3.0 * index, 0.0, 0.0)) for index in range(3)])

    result = build_rna_chain_from_frames(
        library, "UCG", frames, coordinate_unit="angstrom",
        max_translation_residual_A=0.01,
        max_rotation_residual_degrees=0.01,
        max_shared_residue_rmsd_A=0.01,
        max_shared_atom_displacement_A=0.01,
        max_bond_length_deviation_A=0.01,
        include_terminal_suites=True,
    )
    path, assembly = result.path_selection, result.assembly

    assert path.suite_ids == ("UC:5prime", "CG:3prime")
    assert assembly.five_prime_boundary_state == "five_prime_hydroxyl"
    assert assembly.three_prime_boundary_state == "three_prime_hydroxyl"
    assert assembly.five_prime_port is None
    assert assembly.three_prime_port is None
    assert not any(atom["residue_index"] == 0 and atom["atom_name"] == "P"
                   for atom in assembly.atoms)
    trajectory = assembly.to_mdtraj()
    assert trajectory.n_residues == 3
    assert trajectory.n_atoms == len(assembly.atoms)


def test_combined_library_loads_provenance_checked_terminal_suites():
    project_root = Path(__file__).resolve().parents[1]
    library = RNAFragmentLibrary.from_json(
        project_root / "artifacts/rna_pilot/reference/rna_suite_library.json",
        terminal_path=project_root / "artifacts/rna_pilot/reference/rna_terminal_suite_library.json",
    )

    assert library.available_sequences == ("AC", "CG", "GU", "UA")
    assert len(library.get_suites("UC", terminal_role="five_prime")) == 12
    assert len(library.get_suites("GA", terminal_role="three_prime")) == 12


def test_source_contiguous_terminal_path_assembles_one_complete_eight_mer(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    internal_path = project_root / "artifacts/rna_pilot/reference/rna_suite_library.json"
    terminal_path = project_root / "artifacts/rna_pilot/reference/rna_terminal_suite_library.json"
    library = RNAFragmentLibrary.from_json(internal_path, terminal_path=terminal_path)
    internal_data = json.loads(internal_path.read_text(encoding="utf-8"))
    terminal_data = json.loads(terminal_path.read_text(encoding="utf-8"))
    sequence = "UCGUACGA"
    selected_records = [
        next(record for record in terminal_data["suites"]
             if record["source_entry"] == "4U3L" and record["chain_id"] == "A"
             and record["terminal_role"] == "five_prime"),
    ]
    upstream_residue = selected_records[0]["source_residues"][1]
    for step in range(1, len(sequence) - 2):
        pair = sequence[step:step + 2]
        record = next(
            record for record in internal_data["suites"]
            if record["source_entry"] == "4U3L" and record["chain_id"] == "A"
            and record["sequence"] == pair
            and record["source_residues"][0] == upstream_residue
        )
        selected_records.append(record)
        upstream_residue = record["source_residues"][1]
    selected_records.append(next(
        record for record in terminal_data["suites"]
        if record["source_entry"] == "4U3L" and record["chain_id"] == "A"
        and record["terminal_role"] == "three_prime"
        and record["source_residues"][0] == upstream_residue
    ))
    frames = [_identity_frame()]
    for record in selected_records:
        relative = record["downstream_base_frame_relative_to_upstream"]
        current = frames[-1]
        basis = current[1:]
        frames.append(np.vstack((
            current[0] + np.asarray(relative["translation_A"]) @ basis,
            np.asarray(relative["rotation_matrix"]).T @ basis,
        )))
    frames = np.stack(frames)

    result = build_rna_chain_from_frames(
        library, sequence, frames, coordinate_unit="angstrom",
        max_translation_residual_A=1e-5,
        max_rotation_residual_degrees=1e-5,
        max_shared_residue_rmsd_A=1e-5,
        max_shared_atom_displacement_A=1e-5,
        max_bond_length_deviation_A=1e-5,
        include_terminal_suites=True,
    )
    path, assembly = result.path_selection, result.assembly

    assert len(path.placements) == 7
    assert assembly.sequence == sequence
    assert assembly.five_prime_boundary_state == "five_prime_hydroxyl"
    assert assembly.three_prime_boundary_state == "three_prime_hydroxyl"
    assert assembly.shared_residue_max_displacement_A <= 1e-5
    assert assembly.max_bond_length_deviation_A <= 1e-5
    assert all(any(atom["residue_index"] == index and atom["atom_name"] == "O2'"
                   for atom in assembly.atoms) for index in range(len(sequence)))
    trajectory = assembly.to_mdtraj()
    assert trajectory.n_residues == len(sequence)
    assert trajectory.topology.n_bonds == len(assembly.bonds)
    report = result.to_report()
    assert report["schema_version"] == 1
    assert report["target_base_frames_sha256"]
    assert report["source_manifest_sha256"] == library.source_manifest_sha256
    assert report["chemistry_registry_sha256"] == library.chemistry_registry_sha256
    assert report["suite_ids"] == list(path.suite_ids)
    assert report["assembly"]["five_prime_boundary_state"] == "five_prime_hydroxyl"

    output = tmp_path / "terminal_rna_chain.pdb"
    assembly.save_pdb(output)
    reloaded = md.load_pdb(str(output))
    assert reloaded.n_residues == len(sequence)
    assert reloaded.n_atoms == trajectory.n_atoms
    assert reloaded.topology.n_bonds == trajectory.topology.n_bonds


def test_suite_placement_maps_local_atoms_and_reports_zero_frame_residual():
    library = RNAFragmentLibrary([_suite_record()])
    suite = library.get_suites("AC")[0]
    upstream = _identity_frame((10.0, 20.0, 30.0))
    downstream = np.vstack(
        (
            np.array([13.0, 18.0, 34.0]),
            np.eye(3),
        )
    )

    result = library.place_suite(
        suite,
        upstream,
        coordinate_unit="angstrom",
        downstream_frame=downstream,
    )

    downstream_p = next(atom for atom in result.atoms if atom["atom_name"] == "P")
    assert downstream_p["xyz_A"] == pytest.approx((11.6, 22.0, 33.0))
    assert result.downstream_frame == pytest.approx(downstream)
    assert result.translation_residual_A == pytest.approx(0.0)
    assert result.rotation_residual_degrees == pytest.approx(0.0)
    assert result.bonds[0]["atom_1"] == "O3'"


def test_suite_placement_converts_nm_and_reports_translation_and_rotation_residuals():
    library = RNAFragmentLibrary([_suite_record()])
    suite = library.get_suites("AC")[0]
    upstream_nm = _identity_frame((1.0, 2.0, 3.0))
    target_downstream_A = _identity_frame((14.0, 18.0, 34.0))
    target_downstream_A[1:] = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    result = library.place_suite(
        suite,
        upstream_nm,
        coordinate_unit="nm",
        downstream_frame=target_downstream_A,
    )

    upstream_o3 = next(atom for atom in result.atoms if atom["atom_name"] == "O3'")
    assert upstream_o3["xyz_A"] == pytest.approx((10.0, 20.0, 30.0))
    assert result.translation_residual_A == pytest.approx(1.0)
    assert result.rotation_residual_degrees == pytest.approx(90.0)


def test_public_rna_builder_records_inputs_and_rejects_missing_sequence_coverage():
    library = RNAFragmentLibrary([
        _path_suite("AC", "AC:only"),
    ])
    frames = np.stack([_identity_frame(), _identity_frame((3.0, 0.0, 0.0))])
    parameters = {
        "coordinate_unit": "angstrom",
        "max_translation_residual_A": 0.01,
        "max_rotation_residual_degrees": 0.01,
        "max_shared_residue_rmsd_A": 0.1,
        "max_shared_atom_displacement_A": 0.01,
        "max_bond_length_deviation_A": 0.01,
        "include_terminal_suites": False,
    }

    result = build_rna_chain_from_frames(library, "AC", frames, **parameters)
    repeated = build_rna_chain_from_frames(library, "AC", frames.copy(), **parameters)
    changed_frames = frames.copy()
    changed_frames[0, 0, 0] += 0.001
    changed = build_rna_chain_from_frames(library, "AC", changed_frames, **parameters)
    report = result.to_report()

    assert report["algorithm"] == "rna-suite-dp-hard-limits-v1"
    assert report["target_base_frames_sha256"] == repeated.target_base_frames_sha256
    assert report["target_base_frames_sha256"] != changed.target_base_frames_sha256
    assert report["suite_ids"] == ["AC:only"]
    assert report["assembly"]["atom_count"] == len(result.assembly.atoms)
    with pytest.raises(UnsupportedSuiteSequenceError, match="AA"):
        build_rna_chain_from_frames(library, "AA", frames, **parameters)
    with pytest.raises(ValueError, match="must be a bool"):
        build_rna_chain_from_frames(
            library, "AC", frames, **{**parameters, "include_terminal_suites": "yes"}
        )


@pytest.mark.parametrize(
    "frame,unit,message",
    [
        (np.zeros((3, 3)), "angstrom", r"shape \(4, 3\)"),
        (np.vstack((np.zeros(3), np.diag([2.0, 1.0, 1.0]))), "angstrom", "orthonormal"),
        (_identity_frame(), "micron", "coordinate_unit"),
    ],
)
def test_suite_placement_rejects_invalid_frames_and_units(frame, unit, message):
    library = RNAFragmentLibrary([_suite_record()])

    with pytest.raises(ValueError, match=message):
        library.place_suite(
            library.get_suites("AC")[0],
            frame,
            coordinate_unit=unit,
        )


def test_sequence_path_selects_conformers_by_shared_residue_continuity():
    library = RNAFragmentLibrary([
        _path_suite("AC", "AC:discontinuous", shared_residue_offset=0.8),
        _path_suite("AC", "AC:continuous"),
        _path_suite("CG", "CG:continuous"),
    ])
    frames = np.stack([_identity_frame((3.0 * index, 0.0, 0.0)) for index in range(3)])

    result = library.select_suite_path(
        "ACG",
        frames,
        coordinate_unit="angstrom",
        max_translation_residual_A=0.01,
        max_rotation_residual_degrees=0.01,
        max_shared_residue_rmsd_A=0.2,
    )

    assert result.suite_ids == ("AC:continuous", "CG:continuous")
    assert result.shared_residue_rmsd_A == pytest.approx((0.0,))
    assert result.max_translation_residual_A == pytest.approx(0.0)
    assert result.max_rotation_residual_degrees == pytest.approx(0.0)


def test_sequence_path_rejects_candidates_outside_explicit_path_tolerances():
    library = RNAFragmentLibrary([_path_suite("AC", "AC:only", frame_step=3.0)])
    frames = np.stack([_identity_frame((0.0, 0.0, 0.0)),
                       _identity_frame((4.0, 0.0, 0.0))])

    with pytest.raises(NoCompatibleRNAFragmentPathError, match="step 0"):
        library.select_suite_path(
            "AC",
            frames,
            coordinate_unit="angstrom",
            max_translation_residual_A=0.5,
            max_rotation_residual_degrees=1.0,
            max_shared_residue_rmsd_A=0.5,
        )


def test_sequence_path_rejects_discontinuous_adjacent_suite_pair():
    library = RNAFragmentLibrary([
        _path_suite("AC", "AC:offset", shared_residue_offset=0.8),
        _path_suite("CG", "CG:next"),
    ])
    frames = np.stack([_identity_frame((3.0 * index, 0.0, 0.0)) for index in range(3)])

    with pytest.raises(NoCompatibleRNAFragmentPathError, match="shared-residue continuity"):
        library.select_suite_path(
            "ACG",
            frames,
            coordinate_unit="angstrom",
            max_translation_residual_A=0.01,
            max_rotation_residual_degrees=0.01,
            max_shared_residue_rmsd_A=0.2,
        )


def test_suite_library_selects_a_bounded_path_for_a_covered_sequence():
    artifact = Path(__file__).resolve().parents[1] / "artifacts/rna_pilot/reference/rna_suite_library.json"
    library = RNAFragmentLibrary.from_json(artifact)
    sequence = "ACGUA"
    frames = [_identity_frame()]
    for index in range(len(sequence) - 1):
        suite = library.get_suites(sequence[index:index + 2])[0]
        transform = suite["downstream_base_frame_relative_to_upstream"]
        current = frames[-1]
        basis_rows = np.asarray(current[1:])
        rotation = np.asarray(transform["rotation_matrix"])
        translation = np.asarray(transform["translation_A"])
        frames.append(np.vstack((
            current[0] + translation @ basis_rows,
            rotation.T @ basis_rows,
        )))

    result = library.select_suite_path(
        sequence,
        np.stack(frames),
        coordinate_unit="angstrom",
        max_translation_residual_A=2.0,
        max_rotation_residual_degrees=60.0,
        max_shared_residue_rmsd_A=2.0,
    )

    assert len(result.placements) == len(sequence) - 1
    assert tuple(placement.sequence for placement in result.placements) == (
        "AC", "CG", "GU", "UA"
    )
    assert result.max_translation_residual_A <= 2.0
    assert result.max_rotation_residual_degrees <= 60.0
    assert all(value <= 2.0 for value in result.shared_residue_rmsd_A)


def test_compatible_suite_path_assembles_unique_atoms_and_preserves_bond_geometry():
    library = RNAFragmentLibrary([
        _path_suite("AC", "AC:first"),
        _path_suite("CG", "CG:second"),
    ])
    frames = np.stack([_identity_frame((3.0 * index, 0.0, 0.0)) for index in range(3)])
    path = library.select_suite_path(
        "ACG", frames, coordinate_unit="angstrom",
        max_translation_residual_A=0.01,
        max_rotation_residual_degrees=0.01,
        max_shared_residue_rmsd_A=0.2,
    )

    assembly = library.assemble_suite_path(
        path,
        max_shared_atom_displacement_A=1e-6,
        max_bond_length_deviation_A=1e-6,
    )

    assert assembly.sequence == "ACG"
    assert assembly.five_prime_boundary_state == "open_phosphodiester_continuation"
    assert assembly.three_prime_boundary_state == "open_phosphodiester_continuation"
    assert assembly.five_prime_port == (0, "P")
    assert assembly.three_prime_port == (2, "O3'")
    assert len(assembly.atoms) == 15
    assert len(assembly.bonds) == 2
    assert assembly.shared_residue_max_displacement_A == pytest.approx(0.0)
    assert assembly.max_bond_length_deviation_A == pytest.approx(0.0)


def test_assembly_rejects_shared_atom_displacements_above_the_hard_limit():
    library = RNAFragmentLibrary([
        _path_suite("AC", "AC:first"),
        _path_suite("CG", "CG:second", upstream_residue_offset=0.05),
    ])
    frames = np.stack([_identity_frame((3.0 * index, 0.0, 0.0)) for index in range(3)])
    path = library.select_suite_path(
        "ACG", frames, coordinate_unit="angstrom",
        max_translation_residual_A=0.01,
        max_rotation_residual_degrees=0.01,
        max_shared_residue_rmsd_A=0.2,
    )

    with pytest.raises(ValueError, match="shared atom displacement"):
        library.assemble_suite_path(
            path,
            max_shared_atom_displacement_A=0.01,
            max_bond_length_deviation_A=0.1,
        )


def test_source_contiguous_experimental_suite_path_assembles_and_exports(tmp_path):
    artifact = Path(__file__).resolve().parents[1] / "artifacts/rna_pilot/reference/rna_suite_library.json"
    library = RNAFragmentLibrary.from_json(artifact)
    data = json.loads(artifact.read_text(encoding="utf-8"))
    records = [
        record for record in data["suites"]
        if record.get("source_entry") == "4U3L" and record.get("chain_id") == "A"
    ]
    records.sort(key=lambda item: int(item["source_residues"][0]))
    contiguous = next(
        [records[index + offset] for offset in range(3)]
        for index in range(len(records) - 2)
        if all(
            int(records[index + offset]["source_residues"][1])
            == int(records[index + offset + 1]["source_residues"][0])
            for offset in range(2)
        )
    )
    sequence = contiguous[0]["sequence"] + "".join(record["sequence"][1] for record in contiguous[1:])
    frames = [_identity_frame()]
    for record in contiguous:
        transform = record["downstream_base_frame_relative_to_upstream"]
        current = frames[-1]
        basis_rows = current[1:]
        frames.append(np.vstack((
            current[0] + np.asarray(transform["translation_A"]) @ basis_rows,
            np.asarray(transform["rotation_matrix"]).T @ basis_rows,
        )))
    placements = tuple(
        library.place_suite(
            record, frames[index], coordinate_unit="angstrom", downstream_frame=frames[index + 1]
        )
        for index, record in enumerate(contiguous)
    )
    shared_rmsd = tuple(
        library._shared_residue_rmsd_A(placements[index], placements[index + 1])
        for index in range(len(placements) - 1)
    )
    path = RNAFragmentPathSelection(sequence, placements, shared_rmsd, 0.0)

    assembly = library.assemble_suite_path(
        path,
        max_shared_atom_displacement_A=1e-4,
        max_bond_length_deviation_A=1e-4,
    )

    assert assembly.sequence == sequence
    assert assembly.five_prime_boundary_state == "open_phosphodiester_continuation"
    assert assembly.three_prime_boundary_state == "open_phosphodiester_continuation"
    assert assembly.five_prime_port == (0, "P")
    assert assembly.three_prime_port == (len(sequence) - 1, "O3'")
    assert len(assembly.atoms) == len({
        (step + atom["residue_index"], atom["atom_name"])
        for step, placement in enumerate(placements)
        for atom in placement.atoms
    })
    assert assembly.shared_residue_max_displacement_A <= 1e-4
    assert assembly.max_bond_length_deviation_A <= 1e-4
    assembled_coordinates = {
        (atom["residue_index"], atom["atom_name"]): np.asarray(atom["xyz_A"])
        for atom in assembly.atoms
    }
    phosphodiester_lengths = []
    for bond in assembly.bonds:
        atoms = {(bond["residue_index_1"], bond["atom_1"]),
                 (bond["residue_index_2"], bond["atom_2"])}
        if {item[1] for item in atoms} == {"O3'", "P"}:
            phosphodiester_lengths.append(np.linalg.norm(
                assembled_coordinates[next(item for item in atoms if item[1] == "O3'")]
                - assembled_coordinates[next(item for item in atoms if item[1] == "P")]
            ))
    assert len(phosphodiester_lengths) == len(placements)
    assert min(phosphodiester_lengths) >= 1.55
    assert max(phosphodiester_lengths) <= 1.65

    trajectory = assembly.to_mdtraj()
    assert trajectory.n_frames == 1
    assert trajectory.n_residues == len(sequence)
    assert trajectory.n_atoms == len(assembly.atoms)
    assert trajectory.topology.n_bonds == len(assembly.bonds)
    assert [residue.name for residue in trajectory.topology.residues] == list(sequence)
    atom_indices = {
        (atom.residue.index, atom.name): atom.index
        for atom in trajectory.topology.atoms
    }
    for atom_record in assembly.atoms:
        atom_index = atom_indices[(atom_record["residue_index"], atom_record["atom_name"])]
        assert trajectory.xyz[0, atom_index] * 10.0 == pytest.approx(
            atom_record["xyz_A"], abs=1e-6
        )

    output = tmp_path / "internal_rna_segment.pdb"
    from Bio.PDB import MMCIFParser
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict

    cif_path = assembly.save_mmcif(tmp_path / "rna.cif")
    cif = MMCIF2Dict(str(cif_path))
    assert cif["_entity_poly.type"] == ["polyribonucleotide"]
    assert cif["_entity_poly_seq.mon_id"] == list(sequence)
    parsed = MMCIFParser(QUIET=True).get_structure("rna", cif_path)
    parsed_coords = {(a.parent.id[1]-1, a.name): a.coord for a in parsed.get_atoms()}
    assert len(parsed_coords) == len(assembly.atoms)
    for atom in assembly.atoms:
        np.testing.assert_allclose(parsed_coords[atom["residue_index"], atom["atom_name"]],
                                   atom["xyz_A"], atol=2e-6, rtol=0)
    observed = {tuple(sorted(((int(s1)-1, a1), (int(s2)-1, a2))))
                for s1, a1, s2, a2 in zip(cif["_struct_conn.ptnr1_label_seq_id"],
                    cif["_struct_conn.ptnr1_label_atom_id"], cif["_struct_conn.ptnr2_label_seq_id"],
                    cif["_struct_conn.ptnr2_label_atom_id"], strict=True)}
    assert observed == {tuple(sorted(((b["residue_index_1"], b["atom_1"]),
                                      (b["residue_index_2"], b["atom_2"])))) for b in assembly.bonds}
    written = assembly.save_pdb(output)
    assert written == output
    reloaded = md.load_pdb(str(output))
    assert reloaded.n_atoms == trajectory.n_atoms
    assert reloaded.n_residues == trajectory.n_residues
    assert reloaded.topology.n_bonds == trajectory.topology.n_bonds
    expected_bonds = {
        tuple(sorted(((bond["residue_index_1"], bond["atom_1"]),
                      (bond["residue_index_2"], bond["atom_2"]))))
        for bond in assembly.bonds
    }
    reloaded_indices = {
        (atom.residue.index, atom.name): atom.index for atom in reloaded.topology.atoms
    }
    observed_bonds = {
        tuple(sorted(((bond.atom1.residue.index, bond.atom1.name),
                      (bond.atom2.residue.index, bond.atom2.name))))
        for bond in reloaded.topology.bonds
    }
    assert observed_bonds == expected_bonds
    for key, atom_index in atom_indices.items():
        output_index = reloaded_indices[key]
        assert reloaded.xyz[0, output_index] * 10.0 == pytest.approx(
            trajectory.xyz[0, atom_index] * 10.0, abs=0.0011
        )


def test_mdtraj_export_rejects_an_rna_residue_without_o2_prime():
    atoms = tuple(
        {"residue_index": residue_index, "atom_name": atom_name,
         "element": element, "xyz_A": [float(residue_index), float(atom_index), 0.0]}
        for residue_index in range(2)
        for atom_index, (atom_name, element) in enumerate(
            (("P", "P"), ("O3'", "O"), ("C1'", "C"), ("O2'", "O"))
        )
    )
    assembly = RNAFragmentAssembly(
        sequence="AC", atoms=atoms,
        bonds=({"residue_index_1": 0, "atom_1": "O3'", "residue_index_2": 1,
                "atom_2": "P", "order": "single"},),
        source_suite_ids=("test",),
        five_prime_boundary_state="open_phosphodiester_continuation",
        five_prime_port=(0, "P"),
        three_prime_boundary_state="open_phosphodiester_continuation",
        three_prime_port=(1, "O3'"), shared_residue_rmsd_A=(),
        shared_residue_max_displacement_A=0.0, max_bond_length_deviation_A=0.0,
    )
    missing_o2 = replace(
        assembly,
        atoms=tuple(atom for atom in assembly.atoms if not (
            atom["residue_index"] == 1 and atom["atom_name"] == "O2'"
        )),
    )

    with pytest.raises(ValueError, match="standard ribose atom.*O2'"):
        missing_o2.to_mdtraj()
