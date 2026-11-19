import runpy
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATH = runpy.run_path(str(ROOT / 'mdna' / 'path_geometry.py'))
QC = runpy.run_path(str(ROOT / 'mdna' / 'structure_qc.py'))
BENCH = runpy.run_path(str(ROOT / 'scripts' / 'gap_benchmarks' / 'evaluate_open_path_bends.py'))


class TestOpenPath(unittest.TestCase):
    def test_link_torsion_signed_and_missing_atoms_reported(self):
        left = {"C4'": [0, 1, 0], "C3'": [0, 0, 0], "O3'": [1, 0, 0]}
        right = {"P": [1, 0, 1]}
        measure = QC['measure_link_coordinates']
        self.assertAlmostEqual(measure(left, right)['c4_c3_o3_p_deg'], 90.)
        right['P'] = [1, 0, -1]
        self.assertAlmostEqual(measure(left, right)['c4_c3_o3_p_deg'], -90.)
        self.assertIsNone(measure(left, right)['p_o5_c5_deg'])
        right['P'] = [2, 0, 0]
        self.assertIsNone(measure(left, right)['c4_c3_o3_p_deg'])

    def test_fixed_seed_reduces_worst_turn_within_budget(self):
        report = BENCH['benchmark'](True)
        metrics = report['curve']
        self.assertGreater(metrics['max_turn_before_deg'] - metrics['max_turn_after_deg'], 20.)
        self.assertLessEqual(metrics['max_sample_deviation'], report['budget_nm'])
        self.assertLessEqual(metrics['max_control_deviation'], report['budget_nm'])
        self.assertEqual(report['endpoint_displacement_nm'], [0., 0.])
        self.assertIsNone(report['atom_level'])

    def test_zero_budget_preserves_both_points_and_turns(self):
        points = BENCH['controls']()
        result, metrics = PATH['regularize_open_path'](points, max_deviation=0.)
        np.testing.assert_array_equal(result, points)
        self.assertEqual(metrics['max_turn_before_deg'], metrics['max_turn_after_deg'])

    def test_shift_and_rotation_preserve_scores_and_solution(self):
        points = BENCH['controls']()
        first, metrics = PATH['regularize_open_path'](points, max_deviation=0.9)
        rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        shift = [4., -2., 1.]
        second, second_metrics = PATH['regularize_open_path'](points @ rotation + shift, max_deviation=0.9)
        np.testing.assert_allclose(first @ rotation + shift, second, atol=1e-10)
        self.assertAlmostEqual(metrics['max_turn_after_deg'], second_metrics['max_turn_after_deg'])

    def test_rejects_degenerate_path(self):
        points = BENCH['controls']()
        points[1] = points[0]
        with self.assertRaisesRegex(ValueError, 'distinct'):
            PATH['regularize_open_path'](points, max_deviation=1.)
        with self.assertRaisesRegex(ValueError, 'finite'):
            PATH['regularize_open_path'](BENCH['controls'](), max_deviation=float('nan'))

    def test_qc_distinguishes_topology_contact_and_coordinate_geometry(self):
        atoms = [NS(name=n, index=i, element=NS(atomic_number=6)) for i, n in enumerate(
            ["C4'", "C3'", "O3'", 'P', "O5'", "C4'", "C3'"])]
        residues = [NS(index=0, atoms=atoms[:3]), NS(index=1, atoms=atoms[3:])]
        bonds = [(atoms[i], atoms[j]) for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6)]]
        coords_A = [[-3, 0, 0], [-1, 0, 0], [0, 0, 0], [0, 1.6, 0], [1, 1.6, 0],
                    [0, 0, 1.2], [0, 0, 2.3]]
        topology = NS(atoms=atoms, bonds=bonds, chains=[NS(index=0, residues=residues)])
        traj = NS(xyz=np.array([coords_A]) / 10., topology=topology)
        measure = QC['measure_open_chain_geometry']
        result = measure(traj)
        row, = result['links']
        self.assertEqual(result['missing_links'], 0)
        self.assertAlmostEqual(row['o3p_A'], 1.6)
        self.assertAlmostEqual(row['c3_o3_p_deg'], 90.)
        self.assertAlmostEqual(row['o3_p_o5_deg'], 90.)
        # A close pair exists, but the 1.6 Å covalent link is excluded.
        self.assertAlmostEqual(result['min_short_heavy_distance_A'], 1.2)
        topology.bonds = [bond for bond in topology.bonds if bond != (atoms[2], atoms[3])]
        broken = measure(traj)
        self.assertEqual(broken['missing_links'], 1)
        self.assertGreater(broken['short_heavy_contacts'], result['short_heavy_contacts'])
        self.assertEqual(broken['links'][0]['o3p_A'], row['o3p_A'])

    def test_atomic_screen_rejects_unphysical_coordinates_despite_complete_topology(self):
        screen = BENCH['screen_atomic_geometry']
        links = [dict(chain_index=i, o3p_A=d, bond_present=True,
                      c3_o3_p_deg=120., o3_p_o5_deg=105.)
                 for i, d in enumerate((1.6, 7.7))]
        metrics = dict(links=links, missing_links=0, short_heavy_contacts=0, atoms_without_element=0)
        result = screen(metrics)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['failures'], ['out_of_screen_link_length'])
        self.assertEqual([c['long_links'] for c in result['per_chain']], [0, 1])
        links[1]['o3p_A'] = 1.6
        self.assertEqual(screen(metrics)['status'], 'passed_triage_only')


if __name__ == '__main__':
    unittest.main()
