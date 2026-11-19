from pathlib import Path
import runpy
import unittest

import numpy as np
from scipy.spatial.distance import pdist


ROOT = Path(__file__).resolve().parents[1]
PROBE = runpy.run_path(str(ROOT / "scripts/rna_pilot/probe_fixed_anchor_gap.py"))
MEASURE = runpy.run_path(str(ROOT / "mdna/structure_qc.py"))["measure_link_coordinates"]


class TestFixedAnchorGapProbe(unittest.TestCase):
    def test_selection_ignores_truth_and_torsion_metric_wraps(self):
        choose = PROBE["choose_trial"]
        good = {"passed_diagnostic": True, "rmsd_A_and_atom_count": [5., 22]}
        better_truth = {"passed_diagnostic": True, "rmsd_A_and_atom_count": [.1, 22]}
        bad = {"passed_diagnostic": False}
        self.assertEqual(choose(good, better_truth), "rigid")
        self.assertEqual(choose(bad, good), "phosphate")
        self.assertIsNone(choose(bad, bad))
        fields = PROBE["TORSIONS"]
        qc = {"links": [dict.fromkeys(fields, -179.), dict.fromkeys(fields, None)]}
        self.assertEqual(PROBE["torsion_diagnostic"](qc, [dict.fromkeys(fields, 179.)]), [2., None])

    def test_phosphate_turn_preserves_local_bonds_and_respects_cap(self):
        residue = {"P": [0., 0., 0.], "OP1": [-.4, .9, .1], "OP2": [-.3, -.8, -.3],
                   "O5'": [1., 0., 0.], "C5'": [1.5, 1., .2],
                   "C4'": [2., 1.3, 1.], "C3'": [2.3, 1., .1], "O3'": [3., .2, .3]}
        start, middle, end = ({k: np.array(v)+offset for k, v in residue.items()}
                              for offset in (np.zeros(3), [4.5, .5, .4], [9., 0., 0.]))
        target = PROBE["link_matrix"](MEASURE, start, middle, end)
        target[0, 2] += 10
        trial, report = PROBE["fit_missing_residue"](
            MEASURE, start, middle, end, target, max_phosphate_turn_deg=20.)
        self.assertLessEqual(abs(report["phosphate_turn_deg"]), 20.+1e-10)
        self.assertLess(report["final_error"], report["initial_error"])
        for group in (("P", "OP1", "OP2", "O5'", "C5'"), ("C5'", "C4'", "C3'", "O3'")):
            np.testing.assert_allclose(pdist([trial[n] for n in group]),
                                       pdist([middle[n] for n in group]), atol=1e-12)

    def test_fit_preserves_internal_geometry_and_does_not_modify_anchors(self):
        residue = {"P": [0., 0., 0.], "O5'": [1., 0., 0.], "C5'": [1.5, 1., .2],
                   "C4'": [2., 1.3, 1.], "C3'": [2.3, 1., .1], "O3'": [3., .2, .3]}
        start, middle, end = ({k: np.array(v)+offset for k, v in residue.items()}
                              for offset in (np.zeros(3), [4.5, .5, .4], [9., 0., 0.]))
        untouched = [(d, {k: v.copy() for k, v in d.items()}) for d in (start, middle, end)]
        shifted = {k: v+[0., -.12, .05] for k, v in middle.items()}
        target = PROBE["link_matrix"](MEASURE, start, shifted, end)
        trial, report = PROBE["fit_missing_residue"](MEASURE, start, middle, end, target)
        self.assertTrue(report["converged"])
        self.assertLess(report["final_error"], report["initial_error"])
        self.assertLessEqual(report["max_shift_A"], .5)
        names = sorted(middle)
        np.testing.assert_allclose(pdist([trial[n] for n in names]),
                                   pdist([middle[n] for n in names]), atol=1e-12)
        for observed, copy in untouched:
            for name in copy:
                np.testing.assert_array_equal(observed[name], copy[name])


if __name__ == "__main__":
    unittest.main()
