"""Test bounded base-pair frame translations on the fixed synthetic DNA panel."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize

from probe_rigid_phosphate import ALL_STEPS, BUDGET_NM, ROOT, SEQUENCE, controls, probe


def optimize_offsets(anchor_vectors, steps, radii, n_frames, *, frame_cap_A, anchor_budget_A):
    """Move origins only; penalize triangle-inequality reach deficits on both strands."""
    left, right = np.asarray(steps, dtype=int).T
    target = 1.60
    low = np.maximum(0, np.abs(radii - target) - anchor_budget_A)
    high = radii + target + anchor_budget_A

    def unpack(x):
        return np.vstack([np.zeros(3), x.reshape(-1, 3), np.zeros(3)])

    def cost(x):
        offsets = unpack(x)
        v = anchor_vectors + offsets[right] - offsets[left]
        d = np.linalg.norm(v, axis=1)
        excess = d - np.clip(d, low, high)
        change = np.diff(offsets, axis=0)
        value = excess @ excess + .05*np.sum(change**2) + .02*(x @ x)
        g = .04*offsets
        part = 2*excess[:, None]*v / np.maximum(d[:, None], 1e-12)
        np.add.at(g, left, -part)
        np.add.at(g, right, part)
        g[:-1] -= .1*change
        g[1:] += .1*change
        return value, g[1:-1].ravel()

    def constraint(x):
        return frame_cap_A**2 - np.sum(x.reshape(-1, 3)**2, axis=1)

    def jac(x):
        return (-2*np.eye(n_frames-2)[:, :, None]*x.reshape(1, -1, 3)).reshape(n_frames-2, -1)

    initial = np.zeros(3*(n_frames-2))
    result = minimize(cost, initial, jac=True, method="SLSQP",
                      constraints=[dict(type="ineq", fun=constraint, jac=jac)],
                      options=dict(maxiter=150, ftol=1e-9))
    usable = bool(result.success and min(constraint(result.x)) >= -1e-8)
    x = result.x if usable else initial
    return unpack(x), dict(converged=usable, message=str(result.message),
                           objective_before=float(cost(initial)[0]), objective_after=float(cost(x)[0]))


def link_inputs(trajectory, n_frames):
    xyz = trajectory.xyz[0].astype(float)*10
    vectors, steps, radii = [], [], []
    chains = list(trajectory.topology.chains)
    if len(chains) != 2:
        raise ValueError("expected a standard open duplex")
    for strand, chain in enumerate(chains):
        residues = list(chain.residues)
        if len(residues) != n_frames:
            raise ValueError("expected one residue per frame on each strand")
        order = np.arange(n_frames)[::1 if strand == 0 else -1]
        for i, (left, right) in enumerate(zip(residues, residues[1:])):
            pivot = xyz[right.atom("O5'").index]
            vectors.append(pivot - xyz[left.atom("O3'").index])
            radii.append(np.linalg.norm(xyz[right.atom("P").index] - pivot))
            steps.append((order[i], order[i+1]))
    return np.array(vectors), np.array(steps), np.array(radii)


def summarize(result):
    return dict(**result["search_status"], contacts=result["after"]["short_heavy_contacts"],
                over_free_anchor_budget=sum(row["minimum_free_o5_shift_A"] > .5
                                            for row in result["outcomes"]))


def improves_geometry(before, after):
    keys = ("unreachable_links", "contacts", "reachable_links_outside_angle_windows")
    return (all(after[k] <= before[k] for k in keys)
            and any(after[k] < before[k] for k in keys))


def benchmark():
    sys.path.insert(0, str(ROOT))
    from mdna import make

    points = controls()
    straight = np.linspace(points[0], points[-1], len(points))
    cases = {}
    for name, sequence, control, deviation in (
        ("straight_all_steps", ALL_STEPS, straight, None),
        ("straight", SEQUENCE, straight, None),
        ("original", SEQUENCE, points, None),
        ("regularized", SEQUENCE, points, BUDGET_NM),
    ):
        model = make(sequence=sequence, control_points=control, path_deviation_nm=deviation)
        trajectory = model.get_traj()
        n = len(sequence)
        vectors, steps, radii = link_inputs(trajectory, n)
        translation, solver = optimize_offsets(vectors, steps, radii, n, frame_cap_A=.5, anchor_budget_A=.5)
        moved = trajectory[:]
        for strand, chain in enumerate(moved.topology.chains):
            order = range(n) if strand == 0 else reversed(range(n))
            for frame, residue in zip(order, chain.residues, strict=True):
                ids = [atom.index for atom in residue.atoms]
                moved.xyz[0, ids] = trajectory.xyz[0, ids] + translation[frame]/10
        original = probe(trajectory, 1.6, search=True, anchor_budget_A=.5, dual_anchor=True)
        after = probe(moved, 1.6, search=True, anchor_budget_A=.5, dual_anchor=True)
        before_summary, after_summary = summarize(original), summarize(after)
        sweep = []
        for scale in (.25, .5, .75):
            if not np.any(translation):
                break
            trial = trajectory[:]
            trial.xyz[0] = trajectory.xyz[0] + scale*(moved.xyz[0]-trajectory.xyz[0])
            summary = summarize(probe(trial, 1.6, search=True, anchor_budget_A=.5, dual_anchor=True))
            sweep.append(dict(scale=scale, geometry=summary,
                              improves_over_baseline=improves_geometry(before_summary, summary)))
        cases[name] = dict(before=before_summary, after=after_summary, solver=solver,
                           full_shift_improves_over_baseline=improves_geometry(before_summary, after_summary),
                           partial_shift_trials=sweep,
                           frame_offsets_A=translation.tolist(),
                           max_frame_shift_A=float(np.linalg.norm(translation, axis=1).max()),
                           before_outcomes=original["outcomes"], after_outcomes=after["outcomes"])
    return dict(scope="Synthetic DNA diagnostic; orientation held fixed; offsets relative to built path",
                frame_cap_A=.5, anchor_budget_A=.5, cases=cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        report = benchmark()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
