"""Input-contract tests for base-pair frame extraction."""

import json
from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from mdna.geometry import NucleicFrames, calculate_local_helical_parameters


def _trajectory_with_chain_lengths(*lengths):
    topology = md.Topology()
    for chain_index, residue_count in enumerate(lengths):
        chain = topology.add_chain()
        for residue_index in range(residue_count):
            residue = topology.add_residue("A", chain, resSeq=residue_index + 1)
            topology.add_atom("C1'", md.element.carbon, residue)
    return md.Trajectory(xyz=[[[0.0, 0.0, 0.0]] * sum(lengths)], topology=topology)


@pytest.mark.parametrize("chainids", [(0,), (0, 0), (-1, 0), (0, 2)])
def test_nucleic_frames_rejects_invalid_chain_selection(chainids):
    trajectory = _trajectory_with_chain_lengths(2, 2)

    with pytest.raises(ValueError, match="two distinct, valid chain indices"):
        NucleicFrames(trajectory, chainids=chainids)


def test_nucleic_frames_rejects_empty_selected_chain():
    trajectory = _trajectory_with_chain_lengths(0, 2)

    with pytest.raises(ValueError, match="(?i)both selected chains must contain residues"):
        NucleicFrames(trajectory)


def test_nucleic_frames_rejects_unequal_selected_chain_lengths():
    trajectory = _trajectory_with_chain_lengths(2, 3)

    with pytest.raises(ValueError, match="equal residue counts"):
        NucleicFrames(trajectory)


def test_nucleic_frames_extracts_finite_frames_from_standard_ds_rna():
    trajectory = md.load("artifacts/rna_pilot/ideal_A_dsRNA_UCGUACGA.pdb")

    frames = NucleicFrames(trajectory, fit_reference=True)

    assert frames.frames.shape == (8, 1, 4, 3)
    assert frames.step_params.shape == (1, 8, 6)
    assert set(frames.names) == {
        "shear", "stretch", "stagger", "buckle", "propeller", "opening",
        "shift", "slide", "rise", "tilt", "roll", "twist",
    }
    assert np.isfinite(frames.frames).all()
    assert np.isfinite(frames.step_params).all()


def test_nucleic_frames_helical_parameters_match_published_rna_reference_means():
    root = Path("artifacts/rna_pilot/reference/ion_series")
    published = json.loads(
        (root / "x3dna_published_parameters.json").read_text(encoding="utf-8")
    )

    for record in published["parameters"]:
        trajectory = md.load(str(root / f"{record['pdb_id'].lower()}.cif"))
        frames = NucleicFrames(trajectory, fit_reference=False)

        assert frames.helical_parameter_names == ["helical_rise", "helical_twist"]
        parameters, names = frames.get_helical_parameters()
        assert names == frames.helical_parameter_names
        assert np.array_equal(parameters, frames.helical_params)
        assert frames.helical_params.shape == (1, 7, 2)
        observed = frames.helical_params[0].mean(axis=0)
        expected = np.array(
            [record["helical_rise_A_mean"], record["helical_twist_deg_mean"]]
        )
        assert np.allclose(observed, expected, rtol=0.0, atol=(0.03, 0.2))
        assert np.isfinite(frames.helical_params).all()


def test_local_helical_parameters_reject_degenerate_axis():
    one_frame = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )

    with pytest.raises(ValueError, match="Cannot determine local helical axis"):
        calculate_local_helical_parameters(np.stack((one_frame, one_frame)))


def test_local_helical_parameters_reject_non_finite_frame_coordinates():
    one_frame = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    frames = np.stack((one_frame, one_frame.copy()))
    frames[1, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        calculate_local_helical_parameters(frames)


def test_nucleic_frames_single_base_pair_has_empty_step_parameters():
    trajectory = md.load("artifacts/rna_pilot/ideal_A_dsRNA_UCGUACGA.pdb")
    first_chain = next(trajectory.topology.chains)
    second_chain = list(trajectory.topology.chains)[1]
    first_residue = next(first_chain.residues)
    paired_residue = list(second_chain.residues)[-1]
    atom_indices = [
        atom.index
        for residue in (first_residue, paired_residue)
        for atom in residue.atoms
    ]
    single_pair = trajectory.atom_slice(atom_indices)

    frames = NucleicFrames(single_pair, fit_reference=True)

    assert frames.frames.shape == (1, 1, 4, 3)
    assert frames.helical_params.shape == (1, 0, 2)
