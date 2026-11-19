import numpy as np
import pytest

from mdna.nucleic import Connector, Nucleic


def _base_frame(x, angle_degrees):
    angle = np.deg2rad(angle_degrees)
    x_axis = [np.cos(angle), np.sin(angle), 0.0]
    y_axis = [-np.sin(angle), np.cos(angle), 0.0]
    return np.asarray([
        [x, 0.0, 0.0],
        x_axis,
        y_axis,
        [0.0, 0.0, 1.0],
    ])


def _path_frame(x, roll_degrees):
    roll = np.deg2rad(roll_degrees)
    return np.asarray([
        [x, 0.0, 0.0],
        [0.0, -np.cos(roll), -np.sin(roll)],
        [0.0, np.sin(roll), -np.cos(roll)],
        [1.0, 0.0, 0.0],
    ])


def _rotation_angle_degrees(first_basis, second_basis):
    relative_rotation = second_basis @ first_basis.T
    cosine = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def test_default_dna_connector_preserves_endpoints_and_uses_origin_spacing():
    start = Nucleic(sequence="A", frames=_base_frame(0.0, 0.0)[None, ...], circular=False)
    end = Nucleic(sequence="T", frames=_base_frame(2.0, 90.0)[None, ...], circular=False)

    connector = Connector(
        start,
        end,
        sequence="CG",
        n_bp=2,
        frame=0,
        margin=0,
    )

    positions = connector.connected_nuc.frames[:, 0, 0, :]
    step_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)

    assert np.allclose(positions[[0, -1]], [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert np.allclose(step_lengths, np.full(3, 2.0 / 3.0), atol=1e-2)


def test_default_dna_connector_rejects_coincident_endpoint_origins():
    start = Nucleic(sequence="A", frames=_base_frame(0.0, 0.0)[None, ...], circular=False)
    end = Nucleic(sequence="T", frames=_base_frame(0.0, 90.0)[None, ...], circular=False)

    with pytest.raises(ValueError, match="endpoint origins must be separated"):
        Connector(start, end, sequence="CG", n_bp=2, frame=0, margin=0)


def test_default_dna_connector_propagates_start_frame_phase_into_bridge():
    bp_per_turn = 10.5
    twist_per_step = 360.0 / bp_per_turn
    start_roll = 60.0
    start = Nucleic(
        sequence="A",
        frames=_path_frame(0.0, start_roll)[None, ...],
        circular=False,
    )
    end = Nucleic(
        sequence="T",
        frames=_path_frame(2.0, start_roll + 3 * twist_per_step)[None, ...],
        circular=False,
    )

    connector = Connector(start, end, sequence="CG", n_bp=2, frame=0, margin=0)

    frames = connector.connected_nuc.frames[:, 0]
    observed_first_step_twist = _rotation_angle_degrees(frames[0, 1:], frames[1, 1:])

    assert np.isclose(observed_first_step_twist, twist_per_step, atol=1e-3)
