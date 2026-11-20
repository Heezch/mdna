"""Bounded geometry preprocessing for open spatial paths (not atom relaxation)."""

import operator

import numpy as np
from scipy.interpolate import splprep, splev


class StepSpacingError(ValueError):
    """A sampled path did not meet the requested frame-spacing tolerance."""


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


def _endpoint_frames(spline, points, count, spacing, *, min_samples=1001):
    """Use the same arc-length inversion for path screening and frame generation."""
    if count is not None and operator.index(count) < 2:
        raise ValueError("Endpoint preservation requires at least two base pairs.")
    dense_t = np.linspace(0., 1., max(min_samples, 16*(count or 0)))
    dense = np.asarray(splev(dense_t, spline)).T
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
    if not np.isfinite(arc[-1]) or arc[-1] <= 1e-12:
        raise ValueError("Endpoint preservation requires a finite, nonzero path.")
    count = count if count is not None else max(2, round(arc[-1]/spacing)+1)
    target = np.linspace(0., arc[-1], count)
    parameter = np.interp(target, arc, dense_t)
    positions = np.asarray(splev(parameter, spline)).T
    positions[[0, -1]] = points[[0, -1]]
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    if np.any(distances <= 1e-12):
        raise ValueError("Endpoint preservation produced coincident frame origins.")
    deviation = float(np.max(np.abs(distances-spacing)))
    return positions, parameter, target, deviation


def _turns(points):
    steps = np.diff(points, axis=0)
    if np.any(np.linalg.norm(steps, axis=1) <= 1e-12):
        raise ValueError("sampled path contains a degenerate step")
    left, right = steps[:-1], steps[1:]
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(left, right), axis=1),
                                 np.einsum('ij,ij->i', left, right)))


def regularize_open_path(control_points, *, max_deviation, samples=81, iterations=64,
                         frame_count=None, step_spacing=0.34, max_step_deviation=None):
    """Return (control points, diagnostics) for an open cubic path.

    ``max_deviation`` is in input coordinate units. It bounds displacement both
    at control points and at paired samples of equal fractional arc length,
    before any length rescaling. This is a sampled fidelity measure. Endpoints
    remain fixed in these coordinates. No universal angle cutoff is assumed.
    With ``max_step_deviation``, consider only candidates whose frame chord
    lengths meet the input-unit spacing tolerance, ranking them by worst
    turn at the generated frame positions. If the frame count is omitted,
    infer it for each candidate from its sampled length.
    """
    points = _points(control_points)
    budget = float(max_deviation)
    samples, iterations = operator.index(samples), operator.index(iterations)
    if not np.isfinite(budget) or budget < 0 or samples < 4 or iterations < 1:
        raise ValueError("require finite max_deviation >= 0, samples >= 4 and iterations >= 1")
    if np.linalg.norm(points[-1] - points[0]) <= 1e-12:
        raise ValueError("regularize_open_path requires distinct open endpoints")
    if max_step_deviation is not None:
        max_step_deviation = float(max_step_deviation)
        if not np.isfinite(max_step_deviation) or max_step_deviation < 0:
            raise ValueError("max_step_deviation must be finite and nonnegative")
        if not np.isfinite(step_spacing) or step_spacing <= 0:
            raise ValueError("step_spacing must be finite and positive")

    def frame_geometry(controls):
        spline, _ = splprep(controls.T, s=0, k=3)
        frames, _, _, deviation = _endpoint_frames(spline, controls, frame_count, step_spacing)
        return deviation, float(_turns(frames).max(initial=0.))

    baseline, initial_length = _sample(points, samples)
    original_turns = _turns(baseline)
    best = points.copy()
    best_turns = original_turns
    best_length, best_shift, best_control_shift = initial_length, 0., 0.
    if max_step_deviation is None:
        compatible = True
        best_score = float(original_turns.max())
    else:
        initial_deviation, baseline_frame_turn = frame_geometry(points)
        compatible = initial_deviation <= max_step_deviation + 1e-12
        best_score = baseline_frame_turn

    def consider(proposal):
        nonlocal compatible, best, best_turns, best_length, best_shift, best_control_shift, best_score
        control_shift = float(np.linalg.norm(proposal-points, axis=1).max())
        if control_shift > budget + 1e-12:
            return
        try:
            curve, length = _sample(_points(proposal), samples)
            turns = _turns(curve)
            score = float(turns.max())
            if max_step_deviation is not None:
                deviation, score = frame_geometry(proposal)
                if deviation > max_step_deviation + 1e-12:
                    return
        except ValueError:
            return
        shift = float(np.linalg.norm(curve-baseline, axis=1).max())
        if shift > budget or (compatible and score >= best_score - 1e-9):
            return
        best, best_turns, best_length, best_shift = proposal, turns, length, shift
        best_control_shift = control_shift
        best_score = score
        compatible = True

    candidate = points.copy()
    for _ in range(iterations if budget > 0 else 0):
        candidate[1:-1] = (candidate[:-2] + 2*candidate[1:-1] + candidate[2:]) / 4
        offset = candidate - points
        control_shift = float(np.linalg.norm(offset, axis=1).max())
        if control_shift == 0:
            break
        factor = min(1., budget / control_shift)
        # Check smaller moves as well: the spline can overshoot its controls.
        scales = (1., .5, .25) if max_step_deviation is None else (1., .5, .25, .125, .0625, .03125)
        for fraction in scales:
            consider(points + (factor*fraction)*offset)
    if max_step_deviation is not None and budget > 0:
        for _ in range(min(iterations, 3)):
            reference = best
            offsets = (reference[:-2]+reference[2:]-2*reference[1:-1])/4
            for index, offset in enumerate(offsets, 1):
                for fraction in (.0625, .125, .25, .5, 1.):
                    proposal = reference.copy()
                    proposal[index] += fraction*offset
                    consider(proposal)
            if reference is best:
                break
    if not compatible:
        raise StepSpacingError("No sampled endpoint-preserving path satisfies the requested step deviation.")
    metrics = {
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
    if max_step_deviation is not None:
        metrics.update(max_placed_turn_before_deg=baseline_frame_turn, max_placed_turn_after_deg=best_score)
    return best, metrics
