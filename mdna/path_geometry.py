"""Bounded geometry preprocessing for open spatial paths (not atom relaxation)."""

import operator

import numpy as np
from scipy.interpolate import splprep, splev


def _points(value):
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError("control_points must have shape (n, 3) with n >= 4")
    if not np.isfinite(points).all():
        raise ValueError("control_points must be finite")
    if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 1e-12):
        raise ValueError("consecutive control points must be distinct")
    return points


def _sample(points, count):
    spline, _ = splprep(points.T, s=0, k=3)
    dense = np.asarray(splev(np.linspace(0, 1, max(1001, 16 * count)), spline)).T
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    target = np.linspace(0., arc[-1], count)
    sampled = np.column_stack([np.interp(target, arc, axis) for axis in dense.T])
    sampled[[0, -1]] = points[[0, -1]]
    return sampled, float(arc[-1])


def _turns(points):
    steps = np.diff(points, axis=0)
    if np.any(np.linalg.norm(steps, axis=1) <= 1e-12):
        raise ValueError("sampled path contains a degenerate step")
    left, right = steps[:-1], steps[1:]
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(left, right), axis=1),
                                 np.einsum('ij,ij->i', left, right)))


def regularize_open_path(control_points, *, max_deviation, samples=81, iterations=64):
    """Return (control points, diagnostics) for an open cubic path.

    ``max_deviation`` is in input coordinate units. It bounds displacement both
    at control points and at paired samples of equal fractional arc length,
    before any length rescaling. This is a sampled fidelity measure. Endpoints
    remain fixed in these coordinates. No universal angle cutoff is assumed.
    """
    points = _points(control_points)
    budget = float(max_deviation)
    samples, iterations = operator.index(samples), operator.index(iterations)
    if not np.isfinite(budget) or budget < 0 or samples < 4 or iterations < 1:
        raise ValueError("require finite max_deviation >= 0, samples >= 4 and iterations >= 1")
    if np.linalg.norm(points[-1] - points[0]) <= 1e-12:
        raise ValueError("regularize_open_path requires distinct open endpoints")
    baseline, initial_length = _sample(points, samples)
    original_turns = _turns(baseline)
    best = points.copy()
    best_turns = original_turns
    best_length, best_shift, best_control_shift = initial_length, 0., 0.
    candidate = points.copy()
    for _ in range(iterations if budget > 0 else 0):
        candidate[1:-1] = (candidate[:-2] + 2*candidate[1:-1] + candidate[2:]) / 4
        offset = candidate - points
        control_shift = float(np.linalg.norm(offset, axis=1).max())
        if control_shift == 0:
            break
        factor = min(1., budget / control_shift)
        # Check smaller moves as well: the spline can overshoot its controls.
        for scale in (factor, factor / 2, factor / 4):
            proposal = points + scale * offset
            try:
                curve, length = _sample(_points(proposal), samples)
                turns = _turns(curve)
            except ValueError:
                continue
            shift = float(np.linalg.norm(curve - baseline, axis=1).max())
            if shift <= budget and turns.max() < best_turns.max() - 1e-9:
                best, best_turns, best_length, best_shift = proposal, turns, length, shift
                best_control_shift = float(np.linalg.norm(proposal - points, axis=1).max())
    return best, {
        'max_turn_before_deg': float(original_turns.max()),
        'max_turn_after_deg': float(best_turns.max()),
        'p95_turn_before_deg': float(np.percentile(original_turns, 95)),
        'p95_turn_after_deg': float(np.percentile(best_turns, 95)),
        'max_sample_deviation': best_shift,
        'max_control_deviation': best_control_shift,
        'length_before': initial_length,
        'length_after': best_length,
        'samples': samples,
    }
