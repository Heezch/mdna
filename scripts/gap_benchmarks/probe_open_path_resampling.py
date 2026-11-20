"""Choose frame positions on an unchanged curve with bounded chord lengths."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

import numpy as np

from evaluate_open_path_bends import ROOT, SEQUENCE, controls, screen_atomic_geometry


def minimize_max_turn(points, reference, *, spacing, step_limit, displacement_limit):
    """Find the minimax turn on the supplied ordered grid, with fixed endpoints."""
    chord = np.linalg.norm(points[None, :, :]-points[:, None, :], axis=2)
    left, right = np.nonzero(np.triu(np.abs(chord-spacing) <= step_limit, k=1))
    if not len(left):
        return None
    vectors = (points[right]-points[left])/chord[left, right, None]
    previous, following = [], []
    for vertex in range(1, len(points)-1):
        incoming, outgoing = np.flatnonzero(right == vertex), np.flatnonzero(left == vertex)
        previous.extend(np.repeat(incoming, len(outgoing)))
        following.extend(np.tile(outgoing, len(incoming)))
    previous, following = np.asarray(previous, dtype=int), np.asarray(following, dtype=int)
    angles = np.degrees(np.arccos(np.clip(
        np.einsum("ij,ij->i", vectors[previous], vectors[following]), -1., 1.)))
    cost = np.full(len(left), np.inf)
    displacement = np.linalg.norm(reference[:, None, :]-points[None, :, :], axis=2)
    cost[(left == 0) & (displacement[1, right] <= displacement_limit)] = 0.
    trace = []
    for stage in range(2, len(reference)):
        options = np.maximum(cost[previous], angles)
        options[displacement[stage, right[following]] > displacement_limit] = np.inf
        best = np.full(len(left), np.inf)
        np.minimum.at(best, following, options)
        predecessor = np.full(len(left), len(left), dtype=int)
        viable = np.isfinite(options) & (options == best[following])
        np.minimum.at(predecessor, following[viable], previous[viable])
        trace.append(predecessor)
        cost = best
    cost[right != len(points)-1] = np.inf
    edge = int(np.argmin(cost))
    if not np.isfinite(cost[edge]):
        return None
    indices = [int(right[edge]), int(left[edge])]
    for predecessor in reversed(trace):
        edge = predecessor[edge]
        indices.append(int(left[edge]))
    return np.array(indices[::-1])


def evaluate(seed, grid_size, step_limit, displacement_limit):
    sys.path.insert(0, str(ROOT))
    from mdna import make
    from mdna.nucleic import Nucleic
    from mdna.path_geometry import _sample, _turns
    from mdna.spline import SplineFrames
    from mdna.structure_qc import measure_open_chain_geometry

    points = controls(seed)
    grid, _ = _sample(points, grid_size)
    baseline = make(sequence=SEQUENCE, control_points=points, preserve_path_endpoints=True)
    reference = baseline.get_frames()[:, 0, 0, :]
    def measure(model, origins):
        frame_steps = np.linalg.norm(np.diff(origins, axis=0), axis=1)
        qc = measure_open_chain_geometry(model.get_traj())
        return dict(max_turn_deg=float(_turns(origins).max()),
                    max_step_deviation_nm=float(np.abs(frame_steps-.34).max()),
                    max_paired_frame_shift_nm=float(np.linalg.norm(origins-reference, axis=1).max()),
                    endpoint_error_nm=np.linalg.norm(origins[[0, -1]]-points[[0, -1]], axis=1).tolist(),
                    short_heavy_contacts=qc["short_heavy_contacts"], triage=screen_atomic_geometry(qc))

    indices = minimize_max_turn(grid, reference, spacing=.34, step_limit=step_limit,
                                displacement_limit=displacement_limit)
    report = dict(seed=seed, grid_size=grid_size, sequence=SEQUENCE, step_limit_nm=step_limit,
                  displacement_limit_nm=displacement_limit, baseline=measure(baseline, reference))
    if indices is None:
        return dict(report, status="no_feasible_grid_placement")
    placed = grid[indices]
    steps = np.diff(placed, axis=0)
    spline = SplineFrames(points, n_bp=len(placed), preserve_endpoints=True, twist=False)
    spline.positions = placed
    spline.derivatives = np.vstack((steps, steps[-1]))
    spline.frames = []
    spline._compute_frames()
    spline.twist_frames()
    model = Nucleic(sequence=SEQUENCE, n_bp=len(placed), frames=spline.frames, circular=False)
    return dict(report, status="placed", indices=indices.tolist(), frame_origins_nm=placed.tolist(),
                result=measure(model, model.get_frames()[:, 0, 0, :]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--grid-size", type=int, default=1001)
    parser.add_argument("--step-limit", type=float, default=.065)
    parser.add_argument("--displacement-limit", type=float, default=.9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.grid_size < len(SEQUENCE) or not 0. <= args.step_limit < .34
            or not np.isfinite(args.displacement_limit) or args.displacement_limit < 0.):
        parser.error("Use a grid large enough for the sequence, 0 <= step limit < 0.34 and a finite displacement limit.")
    with redirect_stdout(sys.stderr):
        result = evaluate(args.seed, args.grid_size, args.step_limit, args.displacement_limit)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
