"""Checks for the rigid rotation used by the DNA phosphate diagnostic."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "gap_benchmarks"))
from probe_rigid_phosphate import (
    anchor_displacement_bound, candidate_scores, o5_continuous_reach, phosphate_candidates,
    rotate_dual_bounded, rotate_group, rotate_o5_bounded,
    torsion_distance, torsion_vectors,
)
sys.path.pop(0)


class TestRigidPhosphateProbe(unittest.TestCase):
    def test_o3_axis_can_recover_link_unreachable_with_o5_alone(self):
        c3, c4 = np.zeros(3), np.array([0., 0., -1.])
        o3, o5 = np.array([1., 0., 0.]), np.array([1., 2.8, 0.])
        c5, c4_right = o5 + [0, 1, 0], o5 + [0, 2, 0]
        group = o5 + np.eye(3)
        self.assertIsNone(rotate_o5_bounded(c4_right, c5, o5, o3, group, 1.6, .5))
        args = (c4, c3, o3, c4_right, c5, o5, group, 1.6)
        self.assertIsNone(rotate_dual_bounded(*args, .19))
        first, second, result, _, _ = rotate_dual_bounded(*args, .5)
        self.assertLessEqual(np.linalg.norm(first-o3) + np.linalg.norm(second-o5), .5)
        self.assertAlmostEqual(np.linalg.norm(result[0]-first), 1.6)
        self.assertAlmostEqual(np.linalg.norm(first-c3), np.linalg.norm(o3-c3))
        self.assertAlmostEqual((first-c3) @ (c4-c3), (o3-c3) @ (c4-c3))

    def test_torsion_metric_respects_sign_periodicity_and_invalid_geometry(self):
        xyz = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 0, 1],
                        [1, 1, 1], [2, 1, 1], [2, 1, 2]], dtype=float)
        v = torsion_vectors(*xyz)
        self.assertAlmostEqual(v[0, 0], 90.)
        self.assertAlmostEqual(torsion_distance(v, v + 360.)[0], 0.)
        self.assertAlmostEqual(torsion_distance(np.array([[-179.]*4]), np.array([[179.]*4]))[0], 2.)
        v[0, 0] = np.nan
        self.assertTrue(np.isinf(torsion_distance(v, np.zeros((1, 4)))[0]))

    def test_reachable_rotation_preserves_all_pivot_and_intragroup_geometry(self):
        pivot = np.array([2., -1., 4.])
        group = pivot + [[1., 0., 0.], [1., 1., 0.], [1., 0., 1.]]
        upstream = pivot + [0., 2., 0.]
        new, bounds = rotate_group(pivot, upstream, group, 1.6)
        self.assertEqual(bounds, (1., 3.))
        self.assertAlmostEqual(np.linalg.norm(new[0] - upstream), 1.6)
        np.testing.assert_allclose((new-pivot) @ (new-pivot).T,
                                   (group-pivot) @ (group-pivot).T, atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(new-pivot), np.linalg.det(group-pivot))

    def test_unreachable_is_not_reported_as_repaired(self):
        group = np.eye(3)
        new, bounds = rotate_group(np.zeros(3), np.array([8., 0., 0.]), group, 1.6)
        self.assertIsNone(new)
        self.assertEqual(bounds, (7., 9.))

    def test_antiparallel_boundary(self):
        new, _ = rotate_group(np.zeros(3), np.array([2., 0., 0.]), np.eye(3), 3.)
        np.testing.assert_allclose(new[0], [-1., 0., 0.], atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(new), 1.)

    def test_sampled_rotations_keep_link_and_rigid_geometry(self):
        group = np.array([[1., 0., 0.], [1., 1., 0.], [1., 0., 1.]])
        pivot, upstream = np.zeros(3), np.array([0., 2., 0.])
        nearest, _ = rotate_group(pivot, upstream, group, 1.6)
        candidates = phosphate_candidates(pivot, upstream, nearest, grid=8)
        self.assertEqual(candidates.shape, (64, 3, 3))
        np.testing.assert_allclose(np.linalg.norm(candidates[:, 0] - upstream, axis=1), 1.6)
        np.testing.assert_allclose(candidates @ candidates.transpose(0, 2, 1),
                                   np.broadcast_to(group @ group.T, (64, 3, 3)), atol=1e-12)

    def test_contact_scores_respect_topology_exclusion(self):
        group = np.eye(3)
        trial = np.vstack([group, [1., 0., 0.]])
        excluded = np.ones((3, 4), dtype=bool)
        excluded[0, 3] = False
        args = (group[None], trial, np.zeros(3),
                np.array([0., 0., -1.]), np.array([1., 0., -1.]),
                np.array([-1., 0., 0.]), excluded, group)
        self.assertEqual(candidate_scores(*args)[0][0], 1)
        excluded[0, 3] = True
        self.assertEqual(candidate_scores(*args)[0][0], 0)

    def test_anchor_rotation_respects_budget_and_preserves_pivot_geometry(self):
        c4, c5 = np.array([0., 0., -1.]), np.zeros(3)
        pivot, upstream = np.array([1., 0., 0.]), np.array([1., 2.8, 0.])
        group = pivot + np.eye(3)
        _, reach = rotate_group(pivot, upstream, group, 1.6)
        self.assertAlmostEqual(anchor_displacement_bound(reach, 1.6), .2)
        self.assertIsNone(rotate_o5_bounded(c4, c5, pivot, upstream, group, 1.6, .19))
        new, coords, _ = rotate_o5_bounded(c4, c5, pivot, upstream, group, 1.6, .5)
        self.assertLessEqual(np.linalg.norm(new-pivot), .5)
        self.assertAlmostEqual(np.linalg.norm(new-c5), np.linalg.norm(pivot-c5))
        self.assertAlmostEqual((new-c5) @ (c4-c5), (pivot-c5) @ (c4-c5))
        self.assertAlmostEqual(np.linalg.norm(coords[0]-upstream), 1.6)
        self.assertAlmostEqual(np.linalg.norm(coords[0]-new), np.linalg.norm(group[0]-pivot))

    def test_continuous_envelope_with_full_circle_and_fixed_pivot(self):
        c4, c5 = np.array([0., 0., -1.]), np.zeros(3)
        pivot, upstream = np.array([1., 0., 0.]), np.array([2., 0., 0.])
        phosphate = np.array([1.5, 0., 0.])
        np.testing.assert_allclose(o5_continuous_reach(c4, c5, pivot, upstream, phosphate, 0.),
                                   (.5, 1.5))
        np.testing.assert_allclose(o5_continuous_reach(c4, c5, pivot, upstream, phosphate, 2.),
                                   (.5, 3.5))
        np.testing.assert_allclose(o5_continuous_reach(c4, c5, pivot, upstream, phosphate, 1.),
                                   (.5, np.sqrt(3) + .5))

if __name__ == "__main__":
    unittest.main()
