from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/gap_benchmarks"))
from probe_frame_orientation import reorient
sys.path.pop(0)


class FrameOrientationTest(unittest.TestCase):
    def test_fixed_centers_and_end_frames_with_bounded_proper_rotation(self):
        frames = np.tile(np.vstack([np.zeros(3), np.eye(3)]), (3, 1, 1))
        frames[:, 0] = [[0, 0, 0], [0, 0, 1], [1, 0, 1]]
        corrected, angles = reorient(frames, 1, 15.)
        np.testing.assert_array_equal(corrected[:, 0], frames[:, 0])
        np.testing.assert_array_equal(corrected[[0, 2]], frames[[0, 2]])
        np.testing.assert_allclose(angles, [0, 15, 0])
        np.testing.assert_allclose(corrected[1, 1:] @ corrected[1, 1:].T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(corrected[1, 1:]), 1.)
        self.assertGreater(corrected[1, 3, 0], 0.)

    def test_straight_frames_are_unchanged(self):
        frames = np.tile(np.vstack([np.zeros(3), np.eye(3)]), (4, 1, 1))
        frames[:, 0, 2] = np.arange(4)
        result, angles = reorient(frames, 2, 30.)
        np.testing.assert_array_equal(result, frames)
        np.testing.assert_array_equal(angles, np.zeros(4))


if __name__ == "__main__":
    unittest.main()
