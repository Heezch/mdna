"""Input contract tests for coordinate placement on base-pair frames."""

import numpy as np
import pytest

from mdna.generators import StructureGenerator
from mdna import make


def test_endpoint_preserving_make_keeps_requested_nonzero_endpoints():
    points = np.linspace([3., 1., -2.], [4.02, 1., -2.], 4)
    model = make(sequence="ACGT", control_points=points, preserve_path_endpoints=True,
                 max_step_deviation_nm=1e-6)
    origins = model.get_frames()[:, 0, 0, :]
    np.testing.assert_array_equal(origins[[0, -1]], points[[0, -1]])
    np.testing.assert_allclose(np.linalg.norm(np.diff(origins, axis=0), axis=1), .34)


def test_endpoint_preserving_make_rejects_incompatible_sequence_spacing():
    points = np.linspace([0., 0., 0.], [2., 0., 0.], 4)
    with pytest.raises(ValueError, match="step deviation"):
        make(sequence="ACGT", control_points=points, preserve_path_endpoints=True,
             max_step_deviation_nm=.05)


def _frames(n_bp):
    frames = np.tile(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        (n_bp, 1, 1),
    )
    frames[:, 0, 2] = np.arange(n_bp) * 0.34
    return frames


def test_structure_generator_rejects_frame_count_that_differs_from_sequence():
    with pytest.raises(ValueError, match="frame count.*sequence length"):
        StructureGenerator(sequence="ATCG", frames=_frames(3))


@pytest.mark.parametrize("bad_frames", [np.zeros((3, 3)), np.zeros((3, 4, 2))])
def test_structure_generator_rejects_frames_with_invalid_shape(bad_frames):
    with pytest.raises(ValueError, match=r"shape \(n_bp, 4, 3\)"):
        StructureGenerator(sequence="ATC", frames=bad_frames)


def test_structure_generator_rejects_non_finite_target_frames():
    frames = _frames(3)
    frames[1, 0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        StructureGenerator(sequence="ATC", frames=frames)


def test_structure_generator_rejects_scaled_frame_basis():
    frames = _frames(3)
    frames[1, 1] *= 1.5

    with pytest.raises(ValueError, match="orthonormal right-handed"):
        StructureGenerator(sequence="ATC", frames=frames)


def test_structure_generator_rejects_left_handed_frame_basis():
    frames = _frames(3)
    frames[:, 3] *= -1

    with pytest.raises(ValueError, match="orthonormal right-handed"):
        StructureGenerator(sequence="ATC", frames=frames)
