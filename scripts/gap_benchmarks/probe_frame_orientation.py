"""Reestimate tangents at fixed duplex centers on the synthetic DNA panel."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from probe_frame_translation import improves_geometry, summarize
from probe_rigid_phosphate import ALL_STEPS, BUDGET_NM, ROOT, SEQUENCE, controls, probe


def tangent_correction(frames, span, cap_deg):
    """Return bounded minimal rotations toward symmetric chord bisectors."""
    centers = frames[:, 0]
    indices = np.arange(1, len(frames)-1)
    before = centers[indices] - centers[np.maximum(indices-span, 0)]
    after = centers[np.minimum(indices+span, len(frames)-1)] - centers[indices]
    delta = np.zeros_like(centers)
    for i, a, b in zip(indices, before, after, strict=True):
        lengths = np.linalg.norm(a), np.linalg.norm(b)
        if min(lengths) < 1e-12:
            continue
        target = a/lengths[0] + b/lengths[1]
        norm = np.linalg.norm(target)
        if norm < 1e-12:
            continue  # A reversal has no defined bisector.
        target /= norm
        tangent = frames[i, 3]
        axis = np.cross(tangent, target)
        sine = np.linalg.norm(axis)
        if sine > 1e-12:
            angle = np.arctan2(sine, tangent @ target)
            delta[i] = axis / sine * min(angle, np.radians(cap_deg))
    return delta


def reorient(frames, span, cap_deg):
    corrections = tangent_correction(frames, span, cap_deg)
    result = frames.copy()
    result[:, 1:] = np.stack([r.apply(basis) for r, basis in
                               zip(Rotation.from_rotvec(corrections), frames[:, 1:], strict=True)])
    return result, np.degrees(np.linalg.norm(corrections, axis=1))


def benchmark():
    sys.path.insert(0, str(ROOT))
    from mdna import make
    from mdna.generators import StructureGenerator

    points = controls()
    straight = np.linspace(points[0], points[-1], len(points))
    cases = {}
    settings = [(span, cap) for span in (1, 2) for cap in (5., 15., 30.)]
    for name, sequence, control, deviation in (
        ("straight_all_steps", ALL_STEPS, straight, None),
        ("straight", SEQUENCE, straight, None),
        ("original", SEQUENCE, points, None),
        ("regularized", SEQUENCE, points, BUDGET_NM),
    ):
        model = make(sequence=sequence, control_points=control, path_deviation_nm=deviation)
        frames = model.get_frames()[:, 0].copy()

        def score(traj):
            return summarize(probe(traj, 1.6, search=True, anchor_budget_A=.5, dual_anchor=True))

        baseline = score(model.get_traj())
        trials = []
        for span, cap in settings:
            adjusted, correction = reorient(frames, span, cap)
            if not np.any(correction):
                result = baseline
            else:
                traj = StructureGenerator(sequence=sequence, frames=adjusted).get_traj()
                result = score(traj)
            trials.append(dict(span=span, cap_deg=cap, correction_deg=correction.tolist(),
                               max_origin_shift_nm=float(np.linalg.norm(adjusted[:, 0]-frames[:, 0],
                                                                        axis=1).max()),
                               geometry=result, improves=improves_geometry(baseline, result)))
        cases[name] = dict(baseline=baseline, trials=trials)
    return dict(scope="Fixed frame centers and endpoints, synthetic standard DNA only",
                cases=cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        report = benchmark()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
