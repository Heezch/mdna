"""Choose frame positions on an unchanged curve with bounded chord lengths."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
from itertools import product, zip_longest
import json
from pathlib import Path
import sys

import numpy as np

from evaluate_open_path_bends import O3P_SCREEN_A, ROOT, SEQUENCE, controls, screen_atomic_geometry
from screen_open_path_backbone import SHAPE_BUDGETS_NM


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


def frames_on_grid(points, placed, *, tangent_mode="outgoing"):
    from mdna.spline import SplineFrames

    steps = np.diff(placed, axis=0)
    spline = SplineFrames(points, n_bp=len(placed), preserve_endpoints=True, twist=False)
    spline.positions = placed
    if tangent_mode == "bisector":
        directions = steps / np.linalg.norm(steps, axis=1, keepdims=True)
        interior = directions[:-1] + directions[1:]
        if np.any(np.linalg.norm(interior, axis=1) <= 1e-12):
            raise ValueError("Cannot define a symmetric tangent at an exact reversal")
        spline.derivatives = np.vstack((directions[0], interior, directions[-1]))
    elif tangent_mode == "outgoing":
        spline.derivatives = np.vstack((steps, steps[-1]))
    else:
        raise ValueError(f"Unknown tangent mode: {tangent_mode}")
    spline.frames = []
    spline._compute_frames()
    spline.twist_frames()
    return spline.frames


def evaluate(seed, grid_size, step_limit, displacement_limit, *, screen_backbone=False,
             refine_backbone=False, refine_contacts=False, refine_adjacent=False, adjacent_rounds=3):
    sys.path.insert(0, str(ROOT))
    from mdna import make
    from mdna.nucleic import Nucleic
    from mdna.path_geometry import _sample, _turns
    from mdna.structure_qc import measure_open_chain_geometry

    refine_contacts = refine_contacts or refine_adjacent
    points = controls(seed)
    grid, _ = _sample(points, grid_size)
    baseline = make(sequence=SEQUENCE, control_points=points, preserve_path_endpoints=True)
    reference = baseline.get_frames()[:, 0, 0, :]
    def measure(model, origins):
        frame_steps = np.linalg.norm(np.diff(origins, axis=0), axis=1)
        qc = measure_open_chain_geometry(model.get_traj(), include_contact_pairs=refine_contacts)
        return dict(max_turn_deg=float(_turns(origins).max()),
                    max_step_deviation_nm=float(np.abs(frame_steps-.34).max()),
                    max_paired_frame_shift_nm=float(np.linalg.norm(origins-reference, axis=1).max()),
                    endpoint_error_nm=np.linalg.norm(origins[[0, -1]]-points[[0, -1]], axis=1).tolist(),
                    short_heavy_contacts=qc["short_heavy_contacts"], triage=screen_atomic_geometry(qc)), qc

    def build(indices):
        placed = grid[indices]
        frames = frames_on_grid(points, placed)
        model = Nucleic(sequence=SEQUENCE, n_bp=len(placed), frames=frames, circular=False)
        metrics, qc = measure(model, model.get_frames()[:, 0, 0, :])
        return dict(status="placed", indices=indices.tolist(), frame_origins_nm=placed.tolist(),
                    result=metrics), qc

    def place(budget):
        indices = minimize_max_turn(grid, reference, spacing=.34, step_limit=step_limit,
                                    displacement_limit=budget)
        if indices is None:
            return dict(status="no_feasible_grid_placement"), None
        return build(indices)

    baseline_metrics, baseline_qc = measure(baseline, reference)
    report = dict(seed=seed, grid_size=grid_size, sequence=SEQUENCE, step_limit_nm=step_limit,
                  displacement_limit_nm=displacement_limit, baseline=baseline_metrics)
    if not screen_backbone:
        placed, qc = place(displacement_limit)
        if (refine_backbone or refine_contacts) and placed["status"] == "placed":
            frames_by_atom = duplex_atom_frames(baseline.get_traj().topology) if refine_contacts else None
            placed = refine_grid(placed, qc, grid, reference, build, baseline_metrics,
                                 baseline_qc, step_limit, displacement_limit, frames_by_atom=frames_by_atom)
            if refine_adjacent:
                placed = refine_adjacent_frames(placed, grid, reference, build, baseline_metrics,
                                                 baseline_qc, step_limit, displacement_limit,
                                                 max_rounds=adjacent_rounds)
        return dict(report, **placed)

    budgets = [budget for budget in SHAPE_BUDGETS_NM if budget < displacement_limit]
    budgets.append(displacement_limit)
    candidates = []
    for budget in budgets:
        placed, _ = place(budget)
        candidate = dict(search_budget_nm=budget, **placed)
        if candidate["status"] == "placed":
            candidate["failures"] = comparison_failures(candidate["result"], baseline_metrics,
                                                        step_limit, budget)
        candidates.append(candidate)
    accepted = [row for row in candidates if row.get("failures") == []]
    best = min(accepted, key=lambda row: (row["result"]["max_turn_deg"], row["search_budget_nm"]),
               default=None)
    return dict(report, status="candidate_screen", candidates=candidates, selected=best)


def comparison_failures(observed, baseline, step_limit, budget):
    reasons = []
    for key in ("short_links", "long_links", "missing_links", "unmeasured_links"):
        old_count = sum(chain[key] for chain in baseline["triage"]["per_chain"])
        new_count = sum(chain[key] for chain in observed["triage"]["per_chain"])
        if new_count > old_count:
            reasons.append(key)
    if observed["short_heavy_contacts"] > baseline["short_heavy_contacts"]:
        reasons.append("short_heavy_contacts")
    if observed["max_turn_deg"] >= baseline["max_turn_deg"] - 1e-9:
        reasons.append("no_turn_improvement")
    if (any(observed["endpoint_error_nm"]) or observed["max_paired_frame_shift_nm"] > budget + 1e-9
            or observed["max_step_deviation_nm"] > step_limit + 1e-9):
        reasons.append("path_fidelity")
    return reasons


def new_link_outliers(links, baseline_links, n_frames):
    """Return links that crossed either bound, mapped to this duplex's frame order."""
    low, high = O3P_SCREEN_A
    outliers = []
    for index, (link, old) in enumerate(zip(links, baseline_links, strict=True)):
        distance, prior = link["o3p_A"], old["o3p_A"]
        if distance is None or prior is None:
            continue
        if prior <= high < distance or distance < low <= prior:
            step = index % (n_frames - 1)
            # The second chain is written in the opposite direction to the frames.
            left = step if link["chain_index"] == 0 else n_frames - 1 - step
            right = left + 1 if link["chain_index"] == 0 else left - 1
            outliers.append(dict(chain_index=link["chain_index"],
                                 left_residue_index=link["left_residue_index"],
                                 right_residue_index=link["right_residue_index"],
                                 frames=(left, right), baseline_A=prior, observed_A=distance,
                                 excess_A=max(low - distance, distance - high)))
    return sorted(outliers, key=lambda row: -row["excess_A"])


def duplex_atom_frames(topology):
    """Map generated atom indices to frames in the antiparallel duplex."""
    sense, antisense = [list(chain.residues) for chain in topology.chains]
    return {atom.index: frame for frame, pair in enumerate(zip(sense, reversed(antisense), strict=True))
            for residue in pair for atom in residue.atoms}


def new_contact_pairs(qc, baseline_pairs, frames_by_atom):
    return [(i, j, frames_by_atom[i], frames_by_atom[j], distance)
            for i, j, distance in qc["short_heavy_contact_pairs"] if (i, j) not in baseline_pairs]


def refine_grid(start, qc, grid, reference, build, baseline, baseline_qc, step_limit, budget,
                *, frames_by_atom=None):
    """Try single-frame grid shifts near newly out-of-range links, with full atom QC."""
    def score(metrics):
        failures = comparison_failures(metrics, baseline, step_limit, budget)
        links = sum(c["short_links"] + c["long_links"] for c in metrics["triage"]["per_chain"])
        contacts = metrics["short_heavy_contacts"]
        priority = (max(0, contacts - baseline["short_heavy_contacts"]),) if frames_by_atom is not None else ()
        return (len(failures), *priority, links, contacts, metrics["max_turn_deg"])

    n_frames = len(reference)
    baseline_links = baseline_qc["links"]
    start_qc = qc
    baseline_pairs = {(i, j) for i, j, _ in baseline_qc.get("short_heavy_contact_pairs", ())}
    initial_outliers = new_link_outliers(qc["links"], baseline_links, n_frames)
    current, trace, evaluated = start, [], 0
    for _ in range(3):
        current_score = score(current["result"])
        if current_score[0] == 0:
            break
        outliers = new_link_outliers(qc["links"], baseline_links, n_frames)
        link_targets = list(dict.fromkeys(frame for row in outliers for frame in row["frames"]
                                          if 0 < frame < n_frames - 1))
        targets = link_targets
        if frames_by_atom is not None and qc["short_heavy_contacts"] > baseline["short_heavy_contacts"]:
            pairs = new_contact_pairs(qc, baseline_pairs, frames_by_atom)
            pressure = Counter(frame for _, _, frame_i, frame_j, _ in pairs for frame in {frame_i, frame_j}
                               if 0 < frame < n_frames - 1)
            contact_targets = sorted(pressure, key=lambda frame: (-pressure[frame], frame))
            targets = list(dict.fromkeys(frame for pair in zip_longest(contact_targets, link_targets)
                                         for frame in pair if frame is not None))
        indices = np.asarray(current["indices"])
        best, best_score, best_qc, move = current, current_score, qc, None
        for frame in targets[:12]:
            for offset in (-2, -1, 1, 2):
                vertex = indices[frame] + offset
                if not indices[frame-1] < vertex < indices[frame+1]:
                    continue
                steps = np.linalg.norm(grid[vertex] - grid[indices[[frame-1, frame+1]]], axis=1)
                if (np.linalg.norm(grid[vertex] - reference[frame]) > budget
                        or np.any(np.abs(steps - .34) > step_limit)):
                    continue
                trial = indices.copy()
                trial[frame] = vertex
                placed, trial_qc = build(trial)
                evaluated += 1
                metrics = placed["result"]
                if "path_fidelity" in comparison_failures(metrics, baseline, step_limit, budget):
                    continue
                rank = score(metrics)
                if rank < best_score:
                    best, best_score, best_qc, move = placed, rank, trial_qc, (frame, offset)
        if move is None:
            break
        current, qc = best, best_qc
        trace.append(dict(frame=move[0], grid_offset=move[1], score=best_score))
    failures = comparison_failures(current["result"], baseline, step_limit, budget)
    report = dict(status="backbone_refinement", start=start, final=current, failures=failures,
                  evaluated_moves=evaluated, accepted_moves=trace, initial_new_outliers=initial_outliers,
                  final_new_outliers=new_link_outliers(qc["links"], baseline_links, n_frames))
    if frames_by_atom is not None:
        report.update(status="backbone_and_contact_refinement",
                      contact_pair_columns=["atom_i", "atom_j", "frame_i", "frame_j", "distance_A"],
                      initial_new_contact_pairs=new_contact_pairs(start_qc, baseline_pairs, frames_by_atom),
                      final_new_contact_pairs=new_contact_pairs(qc, baseline_pairs, frames_by_atom))
    return report


def refine_adjacent_frames(single, grid, reference, build, baseline, baseline_qc, step_limit, budget,
                           *, max_rounds=3):
    """Move both frames at a newly long link, keeping passing comparison checks."""
    start = single["final"]
    current = start
    baseline_links = baseline_qc["links"]
    n_frames = len(reference)
    initial_failures = single["failures"]
    report = dict(status="adjacent_frame_refinement", single_frame=single,
                  final=current, failures=initial_failures, evaluated_pairs=0, accepted_pairs=[],
                  max_pair_rounds=max_rounds, pair_rounds=[])
    if initial_failures != ["long_links"]:
        return dict(report, pair_search="requires_long_links_as_only_remaining_failure")

    def score(metrics, qc):
        count = sum(chain["long_links"] for chain in metrics["triage"]["per_chain"])
        high = O3P_SCREEN_A[1]
        excess = sum(max(0., row["o3p_A"] - high) for row in qc["links"] if row["o3p_A"] is not None)
        return count, excess, metrics["short_heavy_contacts"], metrics["max_turn_deg"]

    current, current_qc = build(np.asarray(start["indices"]))
    termination = "round_limit"
    for _ in range(max_rounds):
        outliers = new_link_outliers(current_qc["links"], baseline_links, n_frames)
        targets = list(dict.fromkeys(tuple(sorted(row["frames"])) for row in outliers
                                     if row["observed_A"] > O3P_SCREEN_A[1]
                                     and min(row["frames"]) > 0 and max(row["frames"]) < n_frames - 1))
        indices = np.asarray(current["indices"])
        best_score = score(current["result"], current_qc)
        best = None
        round_stats = dict(target_frame_pairs=len(targets), feasible_moves=0, preserved_screen_moves=0)
        report["pair_rounds"].append(round_stats)
        for left, right in targets:
            frames = np.array([left, right])
            for offsets in product((-2, -1, 1, 2), repeat=2):
                vertices = indices[frames] + offsets
                if not indices[left-1] < vertices[0] < vertices[1] < indices[right+1]:
                    continue
                steps = np.linalg.norm(np.diff(grid[[indices[left-1], *vertices, indices[right+1]]], axis=0), axis=1)
                if (np.any(np.linalg.norm(grid[vertices] - reference[frames], axis=1) > budget)
                        or np.any(np.abs(steps - .34) > step_limit)):
                    continue
                trial = indices.copy()
                trial[frames] = vertices
                placed, qc = build(trial)
                report["evaluated_pairs"] += 1
                round_stats["feasible_moves"] += 1
                failures = comparison_failures(placed["result"], baseline, step_limit, budget)
                if any(failure != "long_links" for failure in failures):
                    continue
                round_stats["preserved_screen_moves"] += 1
                rank = score(placed["result"], qc)
                if rank < best_score:
                    best_score = rank
                    best = placed, qc, dict(frames=[left, right], grid_offsets=offsets, score=rank), failures
        if best is None:
            termination = "no_improving_move"
            break
        current, current_qc, move, failures = best
        report["accepted_pairs"].append(move)
        report.update(final=current, failures=failures)
        if not failures:
            termination = "passed_comparison"
            break
    return dict(report, pair_search="evaluated", pair_termination=termination,
                final_new_outliers=new_link_outliers(current_qc["links"], baseline_links, n_frames))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--grid-size", type=int, default=1001)
    parser.add_argument("--step-limit", type=float, default=.065)
    parser.add_argument("--displacement-limit", type=float, default=.9)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--screen-backbone", action="store_true",
                        help="Compare displacement budgets using the placed atoms of each candidate.")
    mode.add_argument("--refine-backbone", action="store_true",
                      help="Try local grid shifts at links crossing the baseline distance window.")
    mode.add_argument("--refine-contacts", action="store_true",
                      help="Target both new short contacts and links crossing the baseline distance window.")
    mode.add_argument("--refine-adjacent", action="store_true",
                      help="After contact refinement, try moving both frames of newly long links.")
    parser.add_argument("--adjacent-rounds", type=int,
                        help="Maximum paired search rounds with --refine-adjacent (default: 3).")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.adjacent_rounds is not None and (not args.refine_adjacent or args.adjacent_rounds < 1):
        parser.error("--adjacent-rounds requires --refine-adjacent and a positive round count.")
    if (args.grid_size < len(SEQUENCE) or not 0. <= args.step_limit < .34
            or not np.isfinite(args.displacement_limit) or args.displacement_limit < 0.):
        parser.error("Use a grid large enough for the sequence, 0 <= step limit < 0.34 and a finite displacement limit.")
    with redirect_stdout(sys.stderr):
        result = evaluate(args.seed, args.grid_size, args.step_limit, args.displacement_limit,
                          screen_backbone=args.screen_backbone, refine_backbone=args.refine_backbone,
                          refine_contacts=args.refine_contacts, refine_adjacent=args.refine_adjacent,
                          adjacent_rounds=args.adjacent_rounds or 3)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
