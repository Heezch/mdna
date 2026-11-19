"""Test rigid reconciliation of overlapping RNA fragment placements."""

import numpy as np
from collections import Counter
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation

from mdna.rna_fragments import RNAFragmentPathSelection
from mdna.structure_qc import measure_open_chain_geometry


def _residue_coordinates(placement, index):
    return {a["atom_name"]: np.asarray(a["xyz_A"]) for a in placement.atoms
            if a["residue_index"] == index}


def overlap_diagnostic(left, right):
    """Describe the mismatch and a necessary displacement bound for reconciliation."""
    first, second = _residue_coordinates(left, 1), _residue_coordinates(right, 0)
    if first.keys() != second.keys() or not first:
        return None
    distances = {name: float(np.linalg.norm(xyz - second[name])) for name, xyz in first.items()}
    groups = {"base": [], "sugar": [], "phosphate": []}
    for name, distance in distances.items():
        group = ("phosphate" if name in {"P", "OP1", "OP2", "OP3"}
                 else "sugar" if "'" in name else "base")
        groups[group].append(distance)
    worst_atom = max(sorted(distances), key=distances.get)
    return dict(
        suite_ids=[left.suite_id, right.suite_id],
        group_rmsd_A={name: float(np.sqrt(np.mean(np.square(values))))
                      for name, values in groups.items() if values},
        worst_atom=worst_atom,
        worst_atom_separation_A=distances[worst_atom],
        min_required_displacement_A=distances[worst_atom] / 2,
    )


def reconcile(first, second, *, reverse_source=False):
    """Fit one intact copy into the overlap, giving each source equal weight."""
    left, right = _residue_coordinates(first, 1), _residue_coordinates(second, 0)
    if reverse_source:
        left, right = right, left
    if left.keys() != right.keys():
        raise ValueError("shared atom inventories differ")
    names = sorted(left)
    source = np.array([left[n] for n in names])
    other = np.array([right[n] for n in names])
    center = source.mean(axis=0)
    target = (source + other)*.5
    result = least_squares(
        lambda x: (Rotation.from_rotvec(x[:3]).apply(source-center)
                   + center + x[3:] - target).ravel(), np.zeros(6), max_nfev=100)
    if not result.success:
        raise ValueError("shared residue rigid fit did not converge")
    moved = Rotation.from_rotvec(result.x[:3]).apply(source-center) + center + result.x[3:]
    shift = max(float(np.linalg.norm(moved-copy, axis=1).max()) for copy in (source, other))
    return dict(zip(names, moved, strict=True)), shift


def fit_joint_poses(placed, consensus, limits, *, frames=None, sequence=None, frame_from_atoms=None,
                    boundaries=(), trace=None, upstream_torsions=0, downstream_angle=None,
                    rotate_downstream_o3=False):
    """Adjust overlap poses together with inequalities for both source copies."""
    source_copies = {}
    links = []
    fixed = {}
    for step, placement in enumerate(placed):
        local = {(a["residue_index"], a["atom_name"]): np.asarray(a["xyz_A"])
                 for a in placement.atoms}
        for (i, name), xyz in local.items():
            key = (step+i, name)
            source_copies.setdefault(key, []).append(xyz)
            fixed[key] = xyz
        for bond in placement.bonds:
            i, j = bond["residue_index_1"], bond["residue_index_2"]
            if i == j:
                continue
            a, b = (i, bond["atom_1"]), (j, bond["atom_2"])
            links.append(((step+i, a[1]), (step+j, b[1]), np.linalg.norm(local[a]-local[b])))
    blocks = []
    for i, coords in enumerate(consensus, 1):
        keys = [(i, n) for n in sorted(coords)]
        pts = np.array([coords[n] for _, n in keys])
        blocks.append((keys, pts, pts.mean(axis=0)))
    center_all = np.mean(list(fixed.values()), axis=0)
    all_keys = list(fixed)
    phosphate_keys = [(0, name) for name in ("P", "OP1", "OP2")] if upstream_torsions else []
    local_keys = phosphate_keys + ([(0, "O5'")] if upstream_torsions == 2 else [])
    torsion_slice = slice(6*len(blocks), 6*len(blocks)+upstream_torsions)
    local_torsion_count = upstream_torsions + int(rotate_downstream_o3)
    o3_parameter = torsion_slice.stop
    local_displacement_cap = .35
    local_rotation_cap = np.deg2rad(20.)
    if phosphate_keys:
        pivot = fixed[0, "O5'"]
        axis = fixed[0, "C5'"] - pivot
        axis = axis / np.linalg.norm(axis)
        phosphate_xyz = np.stack([fixed[key] for key in phosphate_keys])
        radius = np.linalg.norm(np.cross(phosphate_xyz-pivot, axis), axis=1).max()
        if radius > 0. and upstream_torsions == 1:
            displacement_angle = 2*np.arcsin(min(1., (local_displacement_cap-1e-6)/(2*radius)))
            local_rotation_cap = min(local_rotation_cap, displacement_angle)
    if upstream_torsions == 2:
        sugar_pivot = fixed[0, "C5'"]
        sugar_axis = fixed[0, "C4'"] - sugar_pivot
        sugar_axis = sugar_axis / np.linalg.norm(sugar_axis)
    if rotate_downstream_o3:
        end = len(placed)
        o3_key = (end, "O3'")
        o3_pivot = fixed[end, "C3'"]
        o3_axis = fixed[end, "C4'"] - o3_pivot
        o3_axis = o3_axis / np.linalg.norm(o3_axis)
        o3_offset = fixed[o3_key]-o3_pivot
        local_keys.append(o3_key)
    if local_keys:
        local_xyz = np.stack([fixed[key] for key in local_keys])
    if frames is not None:
        from audit_fragment_context import frame_residuals

        reference_frames = []
        for i in range(len(sequence)):
            coords = (consensus[i-1] if 0 < i < len(sequence)-1
                      else {n: v for (j, n), v in fixed.items() if i == j})
            origin, axes = frame_from_atoms(sequence[i], coords)
            reference_frames.append(np.vstack([origin, axes.T]))
        reference_frames = np.stack(reference_frames)

    def pose(x):
        coords = dict(fixed)
        for (keys, pts, center), p in zip(blocks, x[:6*len(blocks)].reshape(-1, 6), strict=True):
            moved = Rotation.from_rotvec(p[:3]).apply(pts-center) + center + p[3:]
            coords.update(zip(keys, moved, strict=True))
        if phosphate_keys:
            angles = x[torsion_slice]
            moved = Rotation.from_rotvec(angles[0]*axis).apply(phosphate_xyz-pivot) + pivot
            if upstream_torsions == 2:
                moved = np.vstack((moved, pivot))
                moved = Rotation.from_rotvec(angles[1]*sugar_axis).apply(moved-sugar_pivot) + sugar_pivot
            coords.update(zip(local_keys[:len(moved)], moved, strict=True))
        if rotate_downstream_o3:
            coords[o3_key] = Rotation.from_rotvec(x[o3_parameter]*o3_axis).apply(o3_offset) + o3_pivot
        return coords

    def global_pose(points, x):
        if not boundaries:
            return points
        return Rotation.from_rotvec(x[-6:-3]).apply(points-center_all) + center_all + x[-3:]

    def geometry(x):
        coords = pose(x)
        shifts = np.array([np.linalg.norm(coords[key] - source)
                           for keys, _, _ in blocks for key in keys for source in source_copies[key]])
        errors = np.array([np.linalg.norm(coords[a]-coords[b])-length for a, b, length in links])
        return shifts, errors

    # Leave a small numerical margin before the independently checked hard limits.
    cap = limits["max_shared_atom_displacement_A"] - 1e-6
    bond_cap = limits["max_bond_length_deviation_A"] - 1e-6
    global_cap = .35
    boundary_cap = .10
    targets = np.array([target for _, _, target in boundaries])

    def boundary_lengths(x, coords):
        return np.array([np.linalg.norm(global_pose(coords[key], x)-xyz)
                         for key, xyz, _ in boundaries])

    def constraint_groups(x, *, linear_margins=False):
        shifts, errors = geometry(x)
        result = {
            "source_atom_displacement_A": cap-shifts if linear_margins else cap**2-shifts**2,
            "internal_bond_length_A": np.r_[bond_cap-errors, bond_cap+errors],
        }
        if boundaries:
            coords = pose(x)
            all_xyz = np.stack([coords[k] for k in all_keys])
            global_shift = global_pose(all_xyz, x)-all_xyz
            errors_boundary = boundary_lengths(x, coords)-targets
            global_shift_squared = np.sum(global_shift**2, axis=1)
            displacement_cap = global_cap-1e-6
            result["global_displacement_A"] = (displacement_cap-np.sqrt(global_shift_squared)
                                                if linear_margins else displacement_cap**2-global_shift_squared)
            result["boundary_bond_length_A"] = np.r_[
                boundary_cap-1e-6-errors_boundary, boundary_cap-1e-6+errors_boundary
            ]
            if local_keys:
                displacement = np.linalg.norm(np.stack([coords[k] for k in local_keys])-local_xyz, axis=1)
                result["local_phosphate_displacement_A"] = local_displacement_cap-1e-6-displacement
            if downstream_angle is not None:
                key, phosphorus, oxygen, low, high = downstream_angle
                left = global_pose(coords[key], x)-phosphorus
                right = oxygen-phosphorus
                angle = np.degrees(np.arctan2(np.linalg.norm(np.cross(left, right)), left @ right))
                result["downstream_angle_degrees"] = np.array([angle-low-1e-6, high-angle-1e-6])
        if frames is None:
            return result
        moved = [reference_frames[0]]
        for frame, (_, _, center), p in zip(reference_frames[1:-1], blocks,
                x[:6*len(blocks)].reshape(-1, 6), strict=True):
            turn = Rotation.from_rotvec(p[:3])
            moved.append(np.vstack([turn.apply(frame[0]-center)+center+p[3:], turn.apply(frame[1:])]))
        moved.append(reference_frames[-1])
        moved = np.stack(moved)
        if boundaries:
            moved[:, 0] = global_pose(moved[:, 0], x)
            moved[:, 1:] = Rotation.from_rotvec(x[-6:-3]).apply(
                moved[:, 1:].reshape(-1, 3)).reshape(-1, 3, 3)
        t, r = frame_residuals(moved, frames)
        result["frame_translation_A"] = limits["max_translation_residual_A"]-1e-6-t
        result["frame_rotation_degrees"] = limits["max_rotation_residual_degrees"]-1e-6-r
        return result

    def constraint(x):
        return np.concatenate(list(constraint_groups(x).values()))

    def cost(x, boundary_index=None):
        shifts, errors = geometry(x)
        coords = pose(x)
        outer_errors = boundary_lengths(x, coords)-targets
        if boundary_index is not None:
            outer_errors = outer_errors[boundary_index:boundary_index+1]
        outer = np.sum(outer_errors**2)
        return float(np.mean(shifts**2) + np.sum(errors**2) + outer + .001*(x @ x))

    start = np.zeros(6*(len(blocks)+bool(boundaries))+local_torsion_count)
    fit_bounds = None
    if local_keys:
        fit_bounds = ([(None, None)] * (6*len(blocks))
                      + [(-local_rotation_cap, local_rotation_cap)] * local_torsion_count
                      + [(None, None)] * 6)
    options = dict(maxiter=100, ftol=1e-10)
    if trace is not None:
        groups = constraint_groups(start)
        stages = [
            ("source_atom_displacement", ("source_atom_displacement_A",)),
            ("internal_geometry", ("internal_bond_length_A",)),
            ("target_frames", ("frame_translation_A", "frame_rotation_degrees")),
            ("external_geometry", ("global_displacement_A", "boundary_bond_length_A",
                                   "local_phosphate_displacement_A", "downstream_angle_degrees")),
        ]
        active = []
        internal_feasible = False
        for stage, names in stages:
            added = [name for name in names if name in groups]
            if not added:
                continue
            active.extend(added)

            def stage_constraint(x):
                measured = constraint_groups(x)
                return np.concatenate([measured[name] for name in active])

            # Test internal geometry in the target frame before moving the fragment.
            stage_bounds = fit_bounds
            if boundaries and stage != "external_geometry":
                stage_bounds = ([(None, None)] * (6 * len(blocks))
                                + [(0.0, 0.0)] * (6+local_torsion_count))
            trial = minimize(cost, start, method="SLSQP", bounds=stage_bounds,
                             constraints=[dict(type="ineq", fun=stage_constraint)], options=options)
            measured = constraint_groups(trial.x, linear_margins=True)
            margins = {name: float(measured[name].min()) for name in active}
            feasible = min(margins.values()) >= -1e-8
            stage_report = dict(stage=stage, feasible=feasible, optimizer_status=int(trial.status),
                                minimum_margin=margins)
            trace.append(stage_report)
            if stage == "external_geometry" and phosphate_keys:
                stage_report["local_phosphate_rotation_degrees"] = float(np.rad2deg(trial.x[torsion_slice][0]))
                if upstream_torsions == 2:
                    stage_report["local_sugar_rotation_degrees"] = float(np.rad2deg(trial.x[torsion_slice][1]))
                if rotate_downstream_o3:
                    stage_report["local_downstream_o3_rotation_degrees"] = float(np.rad2deg(trial.x[o3_parameter]))
                stage_report["trial_boundary_lengths_A"] = boundary_lengths(trial.x, pose(trial.x)).tolist()
            if stage == "target_frames":
                internal_feasible = feasible
            if stage == "external_geometry" and internal_feasible and not phosphate_keys:
                # The end atoms are fixed before the common rigid motion. The
                # distance to a fixed neighbor changes by no more than the atom moves.
                initial = boundary_lengths(start, pose(start))
                endpoints = []
                for i, ((residue_index, atom), _, target) in enumerate(boundaries):
                    needed = max(0., abs(initial[i]-target)-boundary_cap)
                    endpoint = dict(residue_index=residue_index, atom=atom,
                                    initial_length_A=float(initial[i]), target_length_A=float(target),
                                    minimum_atom_motion_A=float(needed),
                                    possible_under_displacement_bound=bool(needed <= global_cap))
                    endpoints.append(endpoint)
                    if needed > global_cap:
                        continue

                    def single_constraint(x):
                        measured = constraint_groups(x)
                        edge = measured["boundary_bond_length_A"]
                        return np.concatenate([
                            *(measured[name] for name in active if name != "boundary_bond_length_A"),
                            edge[[i, i+len(boundaries)]],
                        ])

                    single = minimize(cost, start, args=(i,), method="SLSQP",
                                      constraints=[dict(type="ineq", fun=single_constraint)],
                                      options=options)
                    lengths = boundary_lengths(single.x, pose(single.x))
                    endpoint.update(single_fit_feasible=bool(single_constraint(single.x).min() >= -1e-8),
                                    optimizer_status=int(single.status),
                                    trial_boundary_lengths_A=lengths.tolist())
                stage_report["endpoints"] = endpoints
            if feasible:
                start = trial.x
    seeds = [start]
    if upstream_torsions == 1:
        for fraction in (-.95, -.5, .5, .95):
            seed = start.copy()
            seed[torsion_slice] = fraction * local_rotation_cap
            seeds.append(seed)
    elif upstream_torsions == 2:
        for i, j in ((-.25, 0.), (.25, 0.), (0., -.25), (0., .25), (-.25, .25), (.25, -.25)):
            seed = start.copy()
            seed[torsion_slice] = local_rotation_cap * np.array([i, j])
            seeds.append(seed)
    if rotate_downstream_o3:
        for rotation in (-.25, .25):
            seed = start.copy()
            seed[o3_parameter] = rotation * local_rotation_cap
            seeds.append(seed)
    attempts = []
    for seed in seeds:
        result = minimize(cost, seed, method="SLSQP", bounds=fit_bounds,
                          constraints=[dict(type="ineq", fun=constraint)], options=options)
        if phosphate_keys:
            margins = constraint_groups(result.x, linear_margins=True)
            measured = {name: float(values.min()) for name, values in margins.items()}
            attempt = dict(seed_rotation_degrees=float(np.rad2deg(seed[torsion_slice][0])),
                           final_rotation_degrees=float(np.rad2deg(result.x[torsion_slice][0])),
                           optimizer_status=int(result.status), minimum_margin=measured,
                           trial_boundary_lengths_A=boundary_lengths(result.x, pose(result.x)).tolist())
            if upstream_torsions == 2:
                attempt["seed_sugar_rotation_degrees"] = float(np.rad2deg(seed[torsion_slice][1]))
                attempt["final_sugar_rotation_degrees"] = float(np.rad2deg(result.x[torsion_slice][1]))
            if rotate_downstream_o3:
                attempt["seed_downstream_o3_rotation_degrees"] = float(np.rad2deg(seed[o3_parameter]))
                attempt["final_downstream_o3_rotation_degrees"] = float(np.rad2deg(result.x[o3_parameter]))
            attempts.append(attempt)
        if result.success and constraint(result.x).min() >= -1e-8:
            break
    else:
        if phosphate_keys and trace is not None:
            trace[-1]["final_fit_attempts"] = attempts
        return None
    if phosphate_keys and trace is not None:
        trace[-1]["final_fit_attempts"] = attempts
    coords = pose(result.x)
    return (tuple({n: coords[i, n] for n in block} for i, block in enumerate(consensus, 1)),
            lambda key, points: global_pose(coords[key] if key in local_keys else points, result.x))


def screen_frame_pairs(library, sequence, frames, limits, *, report_frame_misses=False):
    """Return suite placements that meet both original frame tolerances."""
    pairs = []
    frame_misses = []
    translation_limit = limits["max_translation_residual_A"]
    rotation_limit = limits["max_rotation_residual_degrees"]
    for i in range(len(sequence)-1):
        matches = []
        pair = sequence[i:i+2]
        suites = library.get_suites(pair)
        misses = []
        for suite in suites:
            p = library.place_suite(suite, frames[i], coordinate_unit="angstrom",
                                    downstream_frame=frames[i+1], balance_frames=True)
            translation = p.translation_residual_A
            rotation = p.rotation_residual_degrees
            if translation <= translation_limit and rotation <= rotation_limit:
                matches.append(p)
            elif report_frame_misses:
                misses.append((max(translation / translation_limit, rotation / rotation_limit),
                               p.suite_id, translation, rotation))
        if report_frame_misses and not matches:
            closest = min(misses, default=None)
            frame_misses.append(dict(
                step=i, pair=pair, template_count=len(suites),
                closest=None if closest is None else dict(
                    suite_id=closest[1], translation_A=closest[2], rotation_deg=closest[3])))
        pairs.append(matches)
    return pairs, frame_misses


def probe(library, sequence, frames, limits, frame_from_atoms, *, audit_context=None,
          max_paths=10000, reverse_source=False, joint=False, constrain_frames=False, boundaries=(),
          report_frame_misses=False, use_displacement_bound=False, upstream_torsions=0,
          downstream_angle=None, rotate_downstream_o3=False):
    from audit_fragment_context import frame_residuals

    if use_displacement_bound and not joint:
        raise ValueError("the displacement bound requires constrained joint fitting")
    if upstream_torsions not in (0, 1, 2):
        raise ValueError("the upstream torsion count must be 0, 1, or 2")
    if upstream_torsions and (not joint or not any(key == (0, "P") for key, _, _ in boundaries)):
        raise ValueError("terminal phosphate rotation requires an upstream boundary and joint fitting")
    if rotate_downstream_o3 and (upstream_torsions != 2 or downstream_angle is None):
        raise ValueError("downstream O3' rotation requires both upstream torsions and the downstream angle")
    pairs, frame_misses = screen_frame_pairs(library, sequence, frames, limits,
                                             report_frame_misses=report_frame_misses)
    counts = dict(overlap=0, displacement=0, frame=0, assembly=0, contacts=0, passed=0)
    if joint:
        counts["joint_fit_rejected"] = 0
    best = None
    assembly_failures = Counter()
    overlap_misses = {}
    cache = {}

    def paths(selected=(), overlaps=()):
        step = len(selected)
        if step == len(pairs):
            yield selected, overlaps
            return
        for right in pairs[step]:
            if not selected:
                yield from paths((right,), ())
                continue
            left = selected[-1]
            key = step, left.suite_id, right.suite_id
            if key not in cache:
                rmsd = library._shared_residue_rmsd_A(left, right)
                diagnostic = None
                if use_displacement_bound or (report_frame_misses and rmsd is not None
                                              and rmsd > limits["max_shared_residue_rmsd_A"]):
                    diagnostic = overlap_diagnostic(left, right)
                incompatible = rmsd is None or rmsd > limits["max_shared_residue_rmsd_A"]
                if use_displacement_bound:
                    incompatible = (diagnostic is None or diagnostic["min_required_displacement_A"]
                                    > limits["max_shared_atom_displacement_A"])
                value = None
                if incompatible:
                    reason = "overlap"
                    if report_frame_misses and rmsd is not None and diagnostic is not None:
                        previous = overlap_misses.get(step)
                        if previous is None or rmsd < previous["rmsd_A"]:
                            overlap_misses[step] = dict(step=step, rmsd_A=rmsd, **diagnostic)
                else:
                    try:
                        consensus, shift = reconcile(left, right, reverse_source=reverse_source)
                    except ValueError:
                        reason = "assembly"
                    else:
                        reason = ("displacement" if shift > limits["max_shared_atom_displacement_A"]
                                  and not use_displacement_bound else None)
                        if reason is None:
                            value = (rmsd, consensus, shift)
                cache[key] = value, reason
            value, reason = cache[key]
            if reason:
                counts[reason] += 1
                continue
            yield from paths(selected+(right,), overlaps+(value,))

    evaluated = 0
    truncated = False
    joint_traces = []
    # No frame candidate at any step makes the complete path impossible.
    for selected, overlaps in paths() if all(pairs) else ():
        if evaluated >= max_paths:
            truncated = True
            break
        evaluated += 1
        rmsds, consensus, shifts = zip(*overlaps, strict=True) if overlaps else ((), (), ())
        shift = max(shifts, default=0.)
        if joint:
            trace = [] if use_displacement_bound else None
            fitted = fit_joint_poses(selected, consensus, limits,
                                        frames=frames if constrain_frames else None,
                                        sequence=sequence, frame_from_atoms=frame_from_atoms,
                                        boundaries=boundaries, trace=trace,
                                        upstream_torsions=upstream_torsions, downstream_angle=downstream_angle,
                                        rotate_downstream_o3=rotate_downstream_o3)
            if trace is not None and len(joint_traces) < 10:
                joint_traces.append(dict(suite_ids=[p.suite_id for p in selected], stages=trace))
            if fitted is None:
                counts["joint_fit_rejected"] += 1
                continue
            consensus, global_pose = fitted
        # Assemble the original placements with the consensus substituted only
        # in the final coordinates; compare bonds against both unmodified sources.
        path = RNAFragmentPathSelection(sequence, selected, rmsds, 0.)
        try:
            assembly = library.assemble_suite_path(
                path, max_shared_atom_displacement_A=limits["max_shared_atom_displacement_A"],
                max_bond_length_deviation_A=limits["max_bond_length_deviation_A"],
                shared_coordinates={i: coords for i, coords in enumerate(consensus, 1)})
        except ValueError as exc:
            counts["assembly"] += 1
            assembly_failures[str(exc)] += 1
            continue
        terminal_bond_change = None
        if boundaries:
            from dataclasses import replace

            moved = tuple(
                dict(a, xyz_A=global_pose((a["residue_index"], a["atom_name"]),
                                         np.asarray(a["xyz_A"])).tolist()) for a in assembly.atoms)
            if rotate_downstream_o3:
                before = {(a["residue_index"], a["atom_name"]): np.asarray(a["xyz_A"]) for a in assembly.atoms}
                after = {(a["residue_index"], a["atom_name"]): np.asarray(a["xyz_A"]) for a in moved}
                changes = []
                for bond in assembly.bonds:
                    a = (bond["residue_index_1"], bond["atom_1"])
                    b = (bond["residue_index_2"], bond["atom_2"])
                    changes.append(abs(np.linalg.norm(after[a]-after[b])-np.linalg.norm(before[a]-before[b])))
                terminal_bond_change = float(max(changes))
                if terminal_bond_change > 1e-6:
                    counts["assembly"] += 1
                    assembly_failures["terminal rotations changed an internal covalent bond length"] += 1
                    continue
            assembly = replace(assembly, atoms=moved)
        observed = []
        for i, base in enumerate(sequence):
            atoms = {a["atom_name"]: np.asarray(a["xyz_A"]) for a in assembly.atoms
                     if a["residue_index"] == i}
            origin, axes = frame_from_atoms(base, atoms)
            observed.append(np.vstack([origin, axes.T]))
        t, r = frame_residuals(np.stack(observed), frames)
        if (max(t) > limits["max_translation_residual_A"] or
                max(r) > limits["max_rotation_residual_degrees"]):
            counts["frame"] += 1
            continue
        qc = measure_open_chain_geometry(assembly.to_mdtraj())
        if qc["missing_links"] or qc["short_heavy_contacts"]:
            counts["contacts"] += 1
            continue
        counts["passed"] += 1
        if joint:
            shift = assembly.shared_residue_max_displacement_A
        rank = (assembly.max_bond_length_deviation_A, shift, path.suite_ids)
        if best is None or rank < best[0]:
            best = (rank, dict(suites=path.suite_ids, max_shared_shift_A=shift,
                               max_bond_change_A=rank[0], frame_translation_A=t.tolist(),
                               frame_rotation_deg=r.tolist(), qc=qc), assembly)
            if rotate_downstream_o3:
                best[1]["terminal_motion_max_bond_length_change_A"] = terminal_bond_change
                best[1]["candidate_geometry"] = dict(atoms=assembly.atoms, bonds=assembly.bonds)
    if best is not None and audit_context is not None:
        best[1]["context"] = audit_context(best[2])
    report = dict(frame_candidates=list(map(len, pairs)), counts=counts,
                  paths_evaluated=evaluated, search_truncated=truncated,
                  assembly_failures=dict(assembly_failures),
                  best=None if best is None else best[1])
    if report_frame_misses:
        report["frame_misses"] = frame_misses
        report["closest_rejected_overlap_by_step"] = list(overlap_misses.values())
    if use_displacement_bound:
        report["joint_constraint_traces"] = joint_traces
    return report
