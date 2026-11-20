"""Behavior of the bounded frame translation diagnostic."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "gap_benchmarks"))
from probe_frame_translation import improves_geometry, optimize_offsets
sys.path.pop(0)


class TestFrameTranslation(unittest.TestCase):
    def test_keeps_endpoints_and_caps_interior_movement_for_both_strands(self):
        vectors = np.array([[4., 0, 0], [2., 0, 0], [-4., 0, 0], [-2., 0, 0]])
        links = np.array([[0, 1], [1, 2], [1, 0], [2, 1]])
        offsets, report = optimize_offsets(vectors, links, np.full(4, 1.6), 3,
                                           frame_cap_A=.5, anchor_budget_A=.5)
        self.assertTrue(report["converged"])
        self.assertLess(report["objective_after"], report["objective_before"])
        np.testing.assert_array_equal(offsets[[0, -1]], np.zeros((2, 3)))
        self.assertLessEqual(np.linalg.norm(offsets[1]), .500001)
        self.assertLess(offsets[1, 0], 0.)

    def test_rejects_increased_contacts_despite_better_reach(self):
        baseline = dict(unreachable_links=2, contacts=3, reachable_links_outside_angle_windows=1)
        self.assertFalse(improves_geometry(baseline, dict(baseline, unreachable_links=1, contacts=4)))
        self.assertFalse(improves_geometry(baseline, baseline))
        self.assertTrue(improves_geometry(baseline, dict(baseline, unreachable_links=1)))


if __name__ == "__main__":
    unittest.main()
