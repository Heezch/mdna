import numpy as np

import mdna
from mdna.spline import SplineFrames

POINTS = np.array([[0, 0, 0], [1.5, 0.2, 0], [3, 0, 0.2], [4.5, 0.2, 0]], dtype=float)


def test_rescaled_spline_stays_in_place():
    pts = POINTS + [40, -20, 10]
    natural = SplineFrames(control_points=pts).n_bp
    axis = np.asarray(SplineFrames(control_points=pts, n_bp=int(natural * 1.5)).frames)[:, 0]
    assert np.linalg.norm(axis.mean(axis=0) - pts.mean(axis=0)) < 0.5


def largest_linker_step(offset):
    a = mdna.make(sequence="GACGTTCAGC", control_points=POINTS + offset)
    b = mdna.make(sequence="GACGTTCAGC", control_points=POINTS + offset + [8, 0, 0])
    origins = np.asarray(mdna.connect(a, b, n_bp=14, minimize=False).get_frames())[:, -1, 0]
    return np.linalg.norm(np.diff(origins, axis=0), axis=1).max()


def test_connect_does_not_depend_on_where_the_dna_is():
    assert abs(largest_linker_step(np.zeros(3)) - largest_linker_step(np.array([50.0, 0, 0]))) < 0.01
