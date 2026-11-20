"""Test if rigid phosphate rotation about fixed O5' can close DNA links."""
import argparse
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from audit_straight_dna_steps import SEQUENCE as ALL_STEPS
from evaluate_open_path_bends import BUDGET_NM, ROOT, SEQUENCE, controls, screen_atomic_geometry

TORSION_FIELDS = ("c4_c3_o3_p_deg", "c3_o3_p_o5_deg", "o3_p_o5_c5_deg", "p_o5_c5_c4_deg")


def torsion_vectors(c4_left, c3, o3, p, o5, c5, c4_right):
    def dihedral(a, b, c, d):
        axis = c - b
        axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
        x, y = a-b, d-c
        x = x - np.sum(x*axis, axis=-1, keepdims=True)*axis
        y = y - np.sum(y*axis, axis=-1, keepdims=True)*axis
        value = np.degrees(np.arctan2(np.sum(np.cross(axis, x)*y, axis=-1), np.sum(x*y, axis=-1)))
        return np.where((np.linalg.norm(x, axis=-1) > 1e-12) &
                        (np.linalg.norm(y, axis=-1) > 1e-12), value, np.nan)

    return np.column_stack([dihedral(c4_left, c3, o3, p), dihedral(c3, o3, p, o5),
                            dihedral(o3, p, o5, c5), dihedral(p, o5, c5, c4_right)])


def torsion_distance(probe, reference):
    delta = (probe[:, None] - reference + 180.) % 360. - 180.
    distance = np.sqrt(np.mean(delta**2, axis=-1)).min(axis=1)
    return np.where(np.isfinite(distance), distance, np.inf)


def build_reference(rows):
    steps = {row["step"] for row in rows}
    by_step = {s: np.array([[r[k] for k in TORSION_FIELDS] for r in rows if r["step"] == s])
               for s in sorted(steps)}
    return by_step


def phosphate_candidates(pivot, upstream, nearest, grid=24):
    """Sample azimuth about O5'–O3' and roll about O5'–P."""
    theta = np.linspace(0, 2 * np.pi, grid, endpoint=False)
    axis = upstream - pivot
    axis /= np.linalg.norm(axis)
    centered = nearest - pivot
    first = Rotation.from_rotvec(theta[:, None] * axis)
    rolled = Rotation.from_rotvec(theta[:, None] * centered[0] / np.linalg.norm(centered[0]))
    roll_coords = np.stack([r.apply(centered) for r in rolled])
    return np.concatenate([r.apply(roll_coords.reshape(-1, 3)).reshape(-1, 3, 3)
                           for r in first]) + pivot


def candidate_scores(groups, trial, pivot, upstream, c3, c5, excluded, original):
    """Order by contact count, angle-window excess, then displacement."""
    radius = np.max(np.linalg.norm(groups - pivot, axis=2))
    local = np.flatnonzero(np.linalg.norm(trial - pivot, axis=1) < radius + 2.000001)
    allowed = ~excluded[:, local]
    delta = groups[:, :, None, :] - trial[local][None, None, :, :]
    contacts = np.count_nonzero((np.sum(delta * delta, axis=-1) < 4) & allowed, axis=(1, 2))

    def angle(a, b):
        return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1),
                                     np.sum(a*b, axis=-1)))

    p = groups[:, 0]
    values = np.column_stack([angle(c3-upstream, p-upstream), angle(upstream-p, pivot-p),
                              angle(p-pivot, c5-pivot)])
    # Broad diagnostic windows, not conformation-specific validation cutoffs.
    low, high = np.array([100, 90, 95]), np.array([140, 130, 135])
    excess = np.maximum(low-values, 0) + np.maximum(values-high, 0)
    angle_cost = np.sum(excess**2, axis=1)
    movement = np.sum((groups-original)**2, axis=(1, 2))
    return contacts, angle_cost, movement, values


def rotate_group(pivot, upstream, group, target):
    """Return nearest rigid orientation closing the link, or its reach interval."""
    offset = group[0] - pivot
    anchor = upstream - pivot
    r, d = np.linalg.norm(offset), np.linalg.norm(anchor)
    reach = (abs(d - r), d + r)
    if r == 0 or d == 0 or not reach[0] <= target <= reach[1]:
        return None, reach
    u, v = anchor / d, offset / r
    a = float(np.clip(u @ v, -1, 1))
    b = float(np.clip((d*d + r*r - target*target) / (2*d*r), -1, 1))
    perpendicular = v - a*u
    norm = np.linalg.norm(perpendicular)
    if norm < 1e-12:
        basis = np.eye(3)[np.argmin(np.abs(u))]
        perpendicular = basis - (basis @ u)*u
    new_v = b*u + np.sqrt(max(0., 1-b*b))*perpendicular/np.linalg.norm(perpendicular)
    axis = np.cross(v, new_v)
    sine = np.linalg.norm(axis)
    cosine = float(np.clip(v @ new_v, -1, 1))
    if sine < 1e-12:
        if cosine > 0:
            return group.copy(), reach
        axis = np.cross(v, np.eye(3)[np.argmin(np.abs(v))])
    rot = Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.arctan2(sine, cosine))
    return pivot + rot.apply(group - pivot), reach


def anchor_displacement_bound(reach, target):
    """Minimum unrestricted O5' translation, with O3' and P–O5' length fixed."""
    return float(max(reach[0] - target, target - reach[1], 0.))


def o5_continuous_reach(c4, c5, pivot, upstream, phosphate, budget):
    """Exact distance envelope over the budgeted O5' arc and free P orientation."""
    axis = c5 - c4
    length = np.linalg.norm(axis)
    if length < 1e-12:
        raise ValueError("degenerate C4'–C5' axis")
    axis /= length
    center = c5 + ((pivot - c5) @ axis) * axis
    radial = pivot - center
    r = np.linalg.norm(radial)
    anchor = upstream - center
    p_radius = np.linalg.norm(phosphate - pivot)
    if r < 1e-12:
        d_min = d_max = np.linalg.norm(upstream - pivot)
    else:
        limit = 2*np.arcsin(min(1., budget/(2*r)))
        a, b = anchor @ radial, anchor @ np.cross(axis, radial)
        phase = np.arctan2(b, a)
        angles = [-limit, limit]
        for angle in (phase, (phase + 2*np.pi) % (2*np.pi) - np.pi):
            if abs(angle) <= limit:
                angles.append(angle)
        dots = a*np.cos(angles) + b*np.sin(angles)
        ds = np.sqrt(np.maximum(0., anchor @ anchor + r*r - 2*dots))
        d_min, d_max = ds.min(), ds.max()
    return (float(max(d_min-p_radius, p_radius-d_max, 0.)), float(d_max+p_radius))


def rotate_o5_bounded(c4, c5, pivot, upstream, group, target, budget):
    """Sample O5' about C4'–C5'; preserve C5'–O5' and C4'–C5'–O5'."""
    axis = c5 - c4
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        raise ValueError("degenerate C4'–C5' axis")
    angles = np.arange(-180, 180, 1)
    rotations = Rotation.from_rotvec(np.radians(angles)[:, None] * axis / norm)
    pivots = c5 + rotations.apply(pivot - c5)
    shifts = np.linalg.norm(pivots - pivot, axis=1)
    for i in np.argsort(shifts, kind="stable"):
        if shifts[i] > budget:
            break
        rotated = c5 + rotations[i].apply(group - c5)
        new, _ = rotate_group(pivots[i], upstream, rotated, target)
        if new is not None:
            return pivots[i], new, int(angles[i])
    return None


def rotate_dual_bounded(c4_left, c3, upstream, c4_right, c5, pivot, group, target, budget):
    """Find a distance-only closure under a shared O3'+O5' displacement cap."""
    def rotations(a, b):
        axis = b - a
        return Rotation.from_rotvec(np.radians(turns)[:, None] * axis / np.linalg.norm(axis))

    turns = np.arange(-180, 180)
    left_rot, right_rot = rotations(c4_left, c3), rotations(c4_right, c5)
    left = c3 + left_rot.apply(upstream - c3)
    right = c5 + right_rot.apply(pivot - c5)
    dl = np.linalg.norm(left - upstream, axis=1)
    dr = np.linalg.norm(right - pivot, axis=1)
    total = dl[:, None] + dr[None, :]
    pairs = np.argwhere(total <= budget)
    pairs = sorted(pairs, key=lambda ij: total[tuple(ij)])
    r = np.linalg.norm(group[0] - pivot)
    for i, j in pairs:
        d = np.linalg.norm(left[i] - right[j])
        if not abs(d-r) <= target <= d+r:
            continue
        rotated = c5 + right_rot[j].apply(group - c5)
        new, _ = rotate_group(right[j], left[i], rotated, target)
        if new is not None:
            return left[i], right[j], new, int(turns[i]), int(turns[j])
    return None


def probe(trajectory, target, *, search=False, anchor_budget_A=None, torsion_reference=None,
          dual_anchor=False):
    from mdna.structure_qc import measure_open_chain_geometry

    before = measure_open_chain_geometry(trajectory)
    xyz = trajectory.xyz[0].astype(float) * 10
    trial = xyz.copy()
    outcomes = []
    moved = set()
    search_groups = []
    if search:
        atoms = list(trajectory.topology.atoms)
        neighbors = [set() for _ in atoms]
        for a, b in trajectory.topology.bonds:
            neighbors[a.index].add(b.index)
            neighbors[b.index].add(a.index)
        nonheavy = [a.index for a in atoms if a.element is None or a.element.atomic_number <= 1]
    for chain in trajectory.topology.chains:
        residues = list(chain.residues)
        for left, right in zip(residues, residues[1:]):
            ids = [right.atom(name).index for name in ("P", "OP1", "OP2")]
            pivot = xyz[right.atom("O5'").index]
            upstream = xyz[left.atom("O3'").index]
            new, reach = rotate_group(pivot, upstream, xyz[ids], target)
            lower = anchor_displacement_bound(reach, target)
            fixed_reachable = new is not None
            turn = None
            left_turn = None
            left_shift = 0.
            continuous = None
            anchor_shift = 0.
            if not fixed_reachable and anchor_budget_A is not None and lower <= anchor_budget_A:
                c4 = xyz[right.atom("C4'").index]
                c5 = xyz[right.atom("C5'").index]
                continuous = o5_continuous_reach(c4, c5, pivot, upstream, xyz[ids[0]],
                                                 anchor_budget_A)
                result = rotate_o5_bounded(c4, c5, pivot, upstream, xyz[ids], target, anchor_budget_A)
                if result is not None:
                    new_pivot, new, turn = result
                    anchor_shift = float(np.linalg.norm(new_pivot - pivot))
                    pivot_index = right.atom("O5'").index
                    trial[pivot_index] = pivot = new_pivot
                    moved.add(pivot_index)
                elif dual_anchor:
                    c3 = xyz[left.atom("C3'").index]
                    c4_left = xyz[left.atom("C4'").index]
                    dual = rotate_dual_bounded(c4_left, c3, upstream, c4, c5, pivot,
                                               xyz[ids], target, anchor_budget_A)
                    if dual is not None:
                        new_upstream, new_pivot, new, left_turn, turn = dual
                        left_shift = float(np.linalg.norm(new_upstream - upstream))
                        anchor_shift = float(np.linalg.norm(new_pivot - pivot))
                        left_index, right_index = left.atom("O3'").index, right.atom("O5'").index
                        trial[left_index] = upstream = new_upstream
                        trial[right_index] = pivot = new_pivot
                        moved.update((left_index, right_index))
            outcomes.append(dict(chain=chain.index, step=left.name + "-" + right.name,
                                 upstream_residue_index=left.index, reachable=new is not None,
                                 fixed_anchor_reachable=fixed_reachable,
                                 minimum_free_o5_shift_A=lower, proposed_o5_shift_A=anchor_shift,
                                 proposed_o3_shift_A=left_shift, c4_c3_rotation_deg=left_turn,
                                 budgeted_continuous_reach_A=continuous,
                                 c4_c5_rotation_deg=turn,
                                 reach_A=list(reach)))
            if new is not None:
                trial[ids] = new
                moved.update(ids)
                if search:
                    excluded = np.zeros((len(ids), len(xyz)), dtype=bool)
                    excluded[:, ids + nonheavy] = True
                    for row, index in enumerate(ids):
                        reached = {index}
                        for _ in range(3):
                            reached |= {b for a in reached for b in neighbors[a]}
                        excluded[row, list(reached)] = True
                    search_groups.append((len(outcomes)-1, ids, pivot, upstream,
                                          xyz[left.atom("C3'").index],
                                          xyz[right.atom("C5'").index], excluded,
                                          phosphate_candidates(pivot, upstream, new),
                                          xyz[left.atom("C4'").index],
                                          xyz[right.atom("C4'").index]))
    if search:
        # Start from the nearest-rotation baseline; choose against current
        # neighboring groups and retain the incumbent as a candidate.
        for sweep in (search_groups, search_groups[::-1], search_groups):
            for index, ids, pivot, upstream, c3, c5, excluded, grid, _, _ in sweep:
                groups = np.concatenate([trial[ids][None], grid])
                contacts, angles, movement, values = candidate_scores(
                    groups, trial, pivot, upstream, c3, c5, excluded, xyz[ids])
                best = np.lexsort((movement, angles, contacts))[0]
                trial[ids] = groups[best]
                outcomes[index].update(angle_window_excess_deg2=float(angles[best]),
                                       p_o5_c5_deg=float(values[best, 2]))
    if torsion_reference is not None:
        baseline = trial.copy()
        for sweep in (search_groups, search_groups[::-1], search_groups):
            for index, ids, pivot, upstream, c3, c5, excluded, grid, c4_left, c4_right in sweep:
                row = outcomes[index]
                reference = torsion_reference.get(row["step"])
                if reference is None:
                    row["torsion_reference_matches"] = 0
                    continue
                groups = np.concatenate([trial[ids][None], grid])
                contacts, angles, movement, values = candidate_scores(
                    groups, trial, pivot, upstream, c3, c5, excluded, xyz[ids])
                # Preserve the baseline group displacement ceiling and never
                # increase local contacts or angle-window excess in a step.
                cap = np.linalg.norm(baseline[ids]-xyz[ids], axis=1).max()
                eligible = ((np.linalg.norm(groups-xyz[ids], axis=2).max(axis=1) <= cap+1e-10)
                            & (contacts <= contacts[0]) & (angles <= angles[0]+1e-10))
                distance = torsion_distance(
                    torsion_vectors(c4_left, c3, upstream, groups[:, 0], pivot, c5, c4_right),
                    reference)
                valid = np.flatnonzero(eligible & np.isfinite(distance))
                row["torsion_reference_matches"] = len(reference)
                if not len(valid):
                    continue
                best = valid[np.lexsort((movement[valid], distance[valid], contacts[valid]))[0]]
                trial[ids] = groups[best]
                row.update(torsion_reference_rms_deg=float(distance[best]),
                           angle_window_excess_deg2=float(angles[best]), p_o5_c5_deg=float(values[best, 2]))
    candidate = trajectory[:]
    candidate.xyz[0] = trial / 10
    after = measure_open_chain_geometry(candidate)
    displacement = np.linalg.norm((candidate.xyz[0] - trajectory.xyz[0]).astype(float)*10, axis=1)
    reachable_errors = [abs(new["o3p_A"] - target) for new, row in zip(after["links"], outcomes, strict=True)
                        if row["reachable"]]
    triage = screen_atomic_geometry(after)
    unreachable = sum(not row["reachable"] for row in outcomes)
    angle_failures = sum(row.get("angle_window_excess_deg2", 0) > 0 for row in outcomes)
    search_status = None
    if search:
        failures = list(triage["failures"])
        if unreachable:
            failures.append("fixed_anchor_target_unreachable" if anchor_budget_A is None
                            else "target_unreachable_within_sampled_anchor_budget")
        if angle_failures:
            failures.append("outside_probe_angle_windows")
        search_status = dict(status="failed" if failures else "passed_probe_only",
                             failures=failures, unreachable_links=unreachable,
                             reachable_links_outside_angle_windows=angle_failures)
    return dict(before=before, after=after, before_triage=screen_atomic_geometry(before),
                after_triage=triage, search_status=search_status, outcomes=outcomes,
                fixed_atom_max_displacement_A=float(max(d for i, d in enumerate(displacement) if i not in moved)),
                max_atom_displacement_A=float(displacement.max()),
                max_reachable_link_error_A=max(reachable_errors, default=None))


def benchmark(*, search=False, anchor_budget_A=None, torsion_reference=None, dual_anchor=False):
    sys.path.insert(0, str(ROOT))
    from mdna import make

    points = controls()
    straight = np.linspace(points[0], points[-1], len(points))
    target = 1.60  # Round probe target within the existing diagnostic window.
    cases = {}
    for name, sequence, control, deviation in (
        ("straight_all_steps", ALL_STEPS, straight, None),
        ("straight", SEQUENCE, straight, None),
        ("original", SEQUENCE, points, None),
        ("regularized", SEQUENCE, points, BUDGET_NM),
    ):
        model = make(sequence=sequence, control_points=control, path_deviation_nm=deviation)
        cases[name] = probe(model.get_traj(), target, search=search, anchor_budget_A=anchor_budget_A,
                            torsion_reference=torsion_reference, dual_anchor=dual_anchor)
    return dict(target_A=target, cases=cases, search=search, anchor_budget_A=anchor_budget_A,
                dual_anchor=dual_anchor, scope="Geometric probe only; no coordinate output or force field",
                angle_windows_deg={"c3_o3_p": [100, 140], "o3_p_o5": [90, 130],
                                   "p_o5_c5": [95, 135]} if search else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--anchor-budget-A", type=float, help="Optional O5' displacement budget (angstrom)")
    parser.add_argument("--dual-anchor", action="store_true",
                        help="For remaining unreachable links, share the budget between O3' and O5'")
    parser.add_argument("--torsion-reference", type=Path, help="Backbone reference audit JSON")
    args = parser.parse_args()
    if args.anchor_budget_A is not None and (
        not args.search or not np.isfinite(args.anchor_budget_A) or args.anchor_budget_A < 0
    ):
        parser.error("--anchor-budget-A requires --search and a finite nonnegative value")
    if args.dual_anchor and args.anchor_budget_A is None:
        parser.error("--dual-anchor requires --anchor-budget-A")
    reference = holdout = None
    if args.torsion_reference:
        if not args.search:
            parser.error("--torsion-reference requires --search")
        data = args.torsion_reference.read_bytes()
        sources = json.loads(data)["sources"]
        training, evaluation = sources["119D"], sources["1WQY"]
        if training["sha256"] == evaluation["sha256"] or any(
            s["chemistry"] != "DNA" for s in (training, evaluation)
        ):
            parser.error("expected distinct DNA entry sources")
        reference = build_reference(training["links"])
        holdout = build_reference(evaluation["links"])
    with redirect_stdout(sys.stderr):
        report = benchmark(search=args.search, anchor_budget_A=args.anchor_budget_A,
                           torsion_reference=reference, dual_anchor=args.dual_anchor)
    if holdout is not None:
        report["torsion_evaluation"] = dict(
            scoring_entry="119D", evaluation_entry="1WQY", reference_audit_sha256=hashlib.sha256(data).hexdigest(),
            scoring_source_sha256=training["sha256"], evaluation_source_sha256=evaluation["sha256"],
            scoring_contexts=sorted(reference), evaluation_contexts=sorted(holdout),
            cases={name: {stage: evaluate_steps(case[stage]["links"], holdout)
                          for stage in ("before", "after")} for name, case in report["cases"].items()})
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def evaluate_steps(rows, holdout):
    distances, excluded = [], 0
    for row in rows:
        step = f"{row['left_residue_name']}-{row['right_residue_name']}"
        vector = [row[k] for k in TORSION_FIELDS]
        if step in holdout and all(v is not None for v in vector):
            distances.append(float(torsion_distance(np.array([vector]), holdout[step])[0]))
        else:
            excluded += 1
    return dict(measured=len(distances), excluded=excluded,
                median_deg=float(np.median(distances)) if distances else None,
                p90_deg=float(np.percentile(distances, 90)) if distances else None)


if __name__ == "__main__":
    main()
