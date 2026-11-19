from pathlib import Path
import runpy
import unittest

import numpy as np
from scipy.spatial.distance import pdist

AUDIT = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                          "scripts/rna_pilot/audit_fragment_context.py"))


class TestFragmentContext(unittest.TestCase):
    def test_fitting_can_respect_tight_target_frame_budgets(self):
        coords = np.array([[0., 0., 0.], [4., 0., 0.], [2., 1., .2]])
        frames = np.stack([np.vstack([p, np.eye(3)]) for p in coords])
        anchors = np.array([[-1.6, .7, 0.], [5.6, .8, 0.]])
        trial, report = AUDIT["fit_fragment_boundaries"](
            coords, [0, 1], anchors, 1.60, frames=frames,
            target_frames=frames.copy(), frame_limits=(.01, .5))
        self.assertTrue(report["converged"])
        self.assertLessEqual(np.linalg.norm(trial-coords, axis=1).max(), .0100001)
        self.assertLess(np.linalg.norm(np.array(report["after_A"])-1.6),
                        np.linalg.norm(np.array(report["before_A"])-1.6))

    def test_rigid_boundary_fit_preserves_geometry_and_budget(self):
        fit = AUDIT["fit_fragment_boundaries"]
        coords = np.array([[0., 0., 0.], [4., 0., 0.], [2., 1., .2]])
        anchors = np.array([[-1.6, .7, 0.], [5.6, .8, 0.]])
        trial, report = fit(coords, [0, 1], anchors, 1.60)
        self.assertTrue(report["converged"])
        self.assertLessEqual(report["max_shift_A"], .350001)
        self.assertTrue(np.all(np.abs(np.array(report["after_A"])-1.60)
                               < np.abs(np.array(report["before_A"])-1.60)))
        np.testing.assert_allclose(pdist(trial), pdist(coords), atol=1e-12)

    def test_reflection_changes_sign_and_coplanarity_is_unknown(self):
        signs = AUDIT["ribose_signs"]
        atoms = {"C1'": [0, 0, 0], "O4'": [1, 0, 0], "C2'": [0, 1, 0], "N9": [0, 0, 1]}
        self.assertEqual(signs(atoms, "G")["C1'"], 1)
        reflected = {n: np.array(v)*[1, 1, -1] for n, v in atoms.items()}
        self.assertEqual(signs(reflected, "G")["C1'"], -1)
        atoms["N9"] = [1, 1, 0]
        self.assertIsNone(signs(atoms, "G")["C1'"])
        self.assertIsNone(signs(atoms, "G")["C2'"])

    def test_only_explicit_boundary_pair_excluded(self):
        screen = AUDIT["contextual_contacts"]
        atoms = [((0, "P"), [0, 0, 0])]
        env = [("boundary", [1.6, 0, 0]), ("neighbor", [0, 1.7, 0]), ("far", [0, 0, 2.1])]
        output = screen(atoms, env, excluded=[((0, "P"), "boundary")])
        self.assertEqual([r["environment_atom"] for r in output["contacts"]], ["neighbor"])


if __name__ == "__main__":
    unittest.main()
