"""Audit transfer from the ion-series RNA suites to a different synthetic duplex."""
import hashlib
import json
import argparse
from math import prod
import re
import sys

import numpy as np
from audit_fragment_context import contextual_contacts, fit_fragment_boundaries, frame_residuals, ribose_signs

from evaluate_rna_gap_closure_loo import (
    REFERENCE_DIR, _anchor_heavy_atoms, _environment_heavy_atoms, _make_frame, _rmsd,
    _xyz_by_residue, close_rna_gap_from_anchors, gemmi, RNAFragmentLibrary,
)
from mdna import build_rna_chain_from_frames
from mdna.rna_fragments import NoCompatibleRNAFragmentPathError, UnsupportedSuiteSequenceError
from mdna.structure_qc import measure_open_chain_geometry, measure_link_coordinates
from scripts.rna_pilot.extract_rna_suite_library import extract_suite, make_base_frame, validate_suite
from probe_fixed_anchor_gap import FIELDS, TORSIONS, choose_trial, fit_missing_residue, torsion_diagnostic
from probe_shared_residue import overlap_diagnostic, probe as probe_shared_residue, screen_frame_pairs

PATH_LIMITS = dict(max_translation_residual_A=.5, max_rotation_residual_degrees=6.,
                   max_shared_residue_rmsd_A=.5, max_shared_atom_displacement_A=.5,
                   max_bond_length_deviation_A=.1)
GAP_LIMITS = dict(max_translation_residual_A=1., max_rotation_residual_degrees=20.,
                  max_shared_residue_rmsd_A=1.5, max_shared_atom_displacement_A=1.,
                  max_bond_length_deviation_A=.25, max_anchor_atom_displacement_A=1.,
                  max_anchor_connection_length_deviation_A=.25, max_search_states=100_000,
                  max_returned_candidates=25, min_environment_distance_A=1.5)
ANGLE_WINDOWS = dict(c3_o3_p_deg=(100., 140.), o3_p_o5_deg=(90., 130.), p_o5_c5_deg=(95., 135.))


def count_overlap_paths(library, pairs, limits):
    """Count complete paths under each pairwise screen, before reconciliation."""
    if not all(pairs):
        return 0, 0
    rmsd_counts = [1] * len(pairs[0])
    motion_counts = rmsd_counts.copy()
    for left_matches, right_matches in zip(pairs, pairs[1:]):
        next_rmsd = [0] * len(right_matches)
        next_motion = [0] * len(right_matches)
        for i, left in enumerate(left_matches):
            if not (rmsd_counts[i] or motion_counts[i]):
                continue
            for j, right in enumerate(right_matches):
                rmsd = library._shared_residue_rmsd_A(left, right)
                if rmsd is not None and rmsd <= limits["max_shared_residue_rmsd_A"]:
                    next_rmsd[j] += rmsd_counts[i]
                if motion_counts[i]:
                    diagnostic = overlap_diagnostic(left, right)
                    if (diagnostic is not None and diagnostic["min_required_displacement_A"]
                            <= limits["max_shared_atom_displacement_A"]):
                        next_motion[j] += motion_counts[i]
        rmsd_counts, motion_counts = next_rmsd, next_motion
    return sum(rmsd_counts), sum(motion_counts)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_windows(chains):
    rows = [w for chain in chains.values() for w in chain["windows"]]
    covered = [w for w in rows if not w["missing_contexts"]]
    balanced = [w for w in covered if w.get("balanced_path", {}).get("status") == "assembled"]
    joint = [w for w in balanced if (w["balanced_path"]["context"]
             .get("boundary_frame_constrained_trial") or {}).get("passed_joint_frame_distance_diagnostic")]
    return dict(total_windows=len(rows), covered_windows=len(covered),
                baseline_path_windows=sum(isinstance(w["path"], dict) and
                                          w["path"]["status"] == "assembled" for w in covered),
                strict_gap_windows=sum(w["gap"]["returned_candidate_count"] > 0 for w in covered),
                strict_pinned_gap_windows=sum(
                    w.get("gap_strict_exact_anchors", {}).get("returned_candidate_count", 0)>0 for w in covered),
                strict_pinned_angle_gap_windows=sum(
                    w.get("gap_strict_exact_anchors_with_angles", {})
                    .get("returned_candidate_count", 0)>0 for w in covered),
                balanced_path_windows=len(balanced),
                joint_boundary_frame_windows=len(joint))


def audit_assembled_context(assembly, model, window_chain, window, chirality, target_A):
    """Replace only the selected window in the deposited asymmetric unit."""
    removed = {str(r.seqid) for r in window}
    environment = []
    for chain in model:
        for residue in chain:
            seqid = str(residue.seqid)
            if chain.name == window_chain and seqid in removed:
                continue
            for atom in residue:
                if atom.element.atomic_number <= 1 or atom.occ <= 0.:
                    continue
                environment.append(((chain.name, seqid, atom.name, str(atom.altloc)),
                                    [atom.pos.x, atom.pos.y, atom.pos.z]))
    excluded = []
    for env, _ in environment:
        chain, seqid, name, _ = env
        if chain != window_chain:
            continue
        if seqid == str(window[0].seqid.num-1) and name == "O3'":
            excluded.append(((0, "P"), env))
        if seqid == str(window[-1].seqid.num+1) and name == "P":
            excluded.append(((len(window)-1, "O3'"), env))
    by_residue = _xyz_by_residue(assembly)
    signs = [ribose_signs({n: x for (j, n), x in by_residue.items() if i == j}, base)
             for i, base in enumerate(assembly.sequence)]
    contacts = contextual_contacts(list(by_residue.items()), environment, excluded=excluded)
    env_by_id = dict(environment)
    boundaries = [dict(fragment_atom=key, environment_atom=env,
                       distance_A=float(np.linalg.norm(by_residue[key]-env_by_id[env])))
                  for key, env in excluded if key in by_residue]
    repair = None
    constrained = None
    chain_residues = [r for c in model if c.name == window_chain for r in c
                      if r.name in {"A", "C", "G", "U"}]
    order = [str(r.seqid) for r in chain_residues]
    expected = set()
    if str(window[0].seqid) != order[0]:
        expected.add((0, "P"))
    if str(window[-1].seqid) != order[-1]:
        expected.add((len(window)-1, "O3'"))
    if expected and len(boundaries) == len(expected) and {b["fragment_atom"] for b in boundaries} == expected:
        keys = list(by_residue)
        xyz = np.array(list(by_residue.values()))
        indices = [keys.index(b["fragment_atom"]) for b in boundaries]
        fixed = [env_by_id[b["environment_atom"]] for b in boundaries]
        trial, repair = fit_fragment_boundaries(xyz, indices, fixed, target_A)
        def frames_from_coords(coords):
            frames = []
            for i, base in enumerate(assembly.sequence):
                origin, axes = make_base_frame(base, {n: v for (j, n), v in coords.items() if j == i})
                frames.append(np.vstack([origin, axes.T]))
            return np.stack(frames)

        def frame_audit(coords):
            shifts, angles = frame_residuals(frames_from_coords(coords), target_frames)
            return dict(translation_A=shifts.tolist(), rotation_deg=angles.tolist(),
                        passed=bool(max(shifts) <= limits[0]+1e-8 and max(angles) <= limits[1]+1e-8))

        target_frames = np.stack([_make_frame(r) for r in window])
        limits = (PATH_LIMITS["max_translation_residual_A"], PATH_LIMITS["max_rotation_residual_degrees"])
        after_contacts = contextual_contacts(list(zip(keys, trial, strict=True)),
                                             environment, excluded=excluded)["contacts"]
        delta_before = np.abs(np.array(repair["before_A"])-target_A)
        delta_after = np.abs(np.array(repair["after_A"])-target_A)
        transformed = dict(zip(keys, trial, strict=True))
        adjacent = {}
        for chain in model:
            if chain.name == window_chain:
                adjacent.update({str(r.seqid): r for r in chain if str(r.seqid) not in removed})
        before_links, after_links = [], []
        for entry in boundaries:
            index, _ = entry["fragment_atom"]
            neighbor = _anchor_heavy_atoms(adjacent[entry["environment_atom"][1]])
            original = {n: v for (i, n), v in by_residue.items() if i == index}
            moved = {n: v for (i, n), v in transformed.items() if i == index}
            if index == 0:
                before_links.append(measure_link_coordinates(neighbor, original))
                after_links.append(measure_link_coordinates(neighbor, moved))
            else:
                before_links.append(measure_link_coordinates(original, neighbor))
                after_links.append(measure_link_coordinates(moved, neighbor))
        repair.update(
            passed_distance_diagnostic=bool(repair["converged"] and repair["max_shift_A"] <= .350001
                                            and np.all(delta_after <= .10)
                                            and np.all(delta_after <= delta_before)
                                            and len(after_contacts) <= len(contacts["contacts"])),
            after_contacts=after_contacts,
            before_link_geometry=before_links,
            after_link_geometry=after_links,
            diagnostic_distance_tolerance_A=.10)
        repair["before_frame_audit"] = frame_audit(by_residue)
        repair["after_frame_audit"] = frame_audit(transformed)
        repair["passed_joint_frame_distance_diagnostic"] = (
            repair["passed_distance_diagnostic"] and repair["after_frame_audit"]["passed"])

        constrained_xyz, constrained = fit_fragment_boundaries(
            xyz, indices, fixed, target_A, frames=frames_from_coords(by_residue),
            target_frames=target_frames, frame_limits=limits)
        constrained_coords = dict(zip(keys, constrained_xyz, strict=True))
        constrained_links = []
        for entry in boundaries:
            index, _ = entry["fragment_atom"]
            neighbor = _anchor_heavy_atoms(adjacent[entry["environment_atom"][1]])
            moved = {n: v for (i, n), v in constrained_coords.items() if i == index}
            constrained_links.append(measure_link_coordinates(neighbor, moved) if index == 0
                                     else measure_link_coordinates(moved, neighbor))
        constrained["after_link_geometry"] = constrained_links
        constrained["boundary_angles_passed"] = all(
            link[name] is not None and lo <= link[name] <= hi
            for link in constrained_links for name, (lo, hi) in ANGLE_WINDOWS.items())
        constrained["after_frame_audit"] = frame_audit(constrained_coords)
        constrained["after_contacts"] = contextual_contacts(
            list(constrained_coords.items()), environment, excluded=excluded)["contacts"]
        errors = np.abs(np.array(constrained["after_A"])-target_A)
        constrained["passed_joint_frame_distance_diagnostic"] = bool(
            constrained["converged"] and constrained["max_shift_A"] <= .350001
            and constrained["after_frame_audit"]["passed"] and np.all(errors <= .10)
            and np.all(errors <= delta_before)
            and len(constrained["after_contacts"]) <= len(contacts["contacts"]))
        constrained["passed_context_diagnostic"] = bool(
            constrained["passed_joint_frame_distance_diagnostic"]
            and constrained["boundary_angles_passed"] and not constrained["after_contacts"]
            and all(sign == chirality and None not in sign.values() for sign in signs))
    return dict(ribose_signs=signs, reference_ribose_signs=chirality,
                all_ribose_signs_match=all(sign == chirality and None not in sign.values() for sign in signs),
                boundary_distances=boundaries,
                boundary_initial_geometry=repair["before_link_geometry"] if repair else [],
                expected_boundary_count=len(expected),
                boundary_rigid_trial=repair,
                boundary_frame_constrained_trial=constrained,
                **contacts)


def evaluate(entry="1RNA", supplement=None, *, sensitivity=False, long_paths_only=False, progress=False,
             frame_preflight=False, window_filter=None):
    if frame_preflight:
        long_paths_only = True
    if window_filter is not None and not long_paths_only:
        raise ValueError("window filtering requires a long-path evaluation")
    source = REFERENCE_DIR / f"{entry.lower()}.cif"
    target_hash = digest(source)
    internal = REFERENCE_DIR / "rna_suite_library.json"
    terminal = REFERENCE_DIR / "rna_terminal_suite_library.json"
    library = RNAFragmentLibrary.from_json(internal, terminal_path=terminal)
    metadata = json.loads(internal.read_bytes())
    terminal_metadata = json.loads(terminal.read_bytes())
    if any(target_hash == row["sha256"] or row["file"].lower().startswith(f"{entry.lower()}.")
           for row in metadata["sources"]):
        raise ValueError("held-out target must stay out of the template library")
    if any(row["suite_id"].upper().startswith(f"{entry}:")
           for doc in (metadata, terminal_metadata) for row in doc["suites"]):
        raise ValueError("held-out suite found in the template library")
    extra_sources = []
    manifest = None
    if supplement:
        source_entries = (supplement,) if isinstance(supplement, str) else supplement
        source_hashes = {row["sha256"] for row in metadata["sources"]}
        supplements = []
        for template_entry in source_entries:
            extra_path = REFERENCE_DIR / f"{template_entry.lower()}.cif"
            extra_hash = digest(extra_path)
            if entry.upper() == template_entry.upper() or extra_hash == target_hash:
                raise ValueError("target cannot be a template source")
            if extra_hash in source_hashes:
                raise ValueError(f"duplicate template coordinates: {template_entry}")
            source_hashes.add(extra_hash)
            start_count = len(supplements)
            for chain in gemmi.read_structure(str(extra_path))[0]:
                residues = [r for r in chain if r.name in {"A", "C", "G", "U"}]
                sequence = "".join(r.name for r in residues)
                for i in range(1, len(residues) - 2):
                    if residues[i+1].seqid.num != residues[i].seqid.num + 1:
                        continue
                    suite = extract_suite(template_entry.lower(), chain.name, residues[i], residues[i+1],
                                          extra_hash, sequence)
                    validate_suite(suite)
                    supplements.append(suite)
            extra_sources.append(dict(entry=template_entry, sha256=extra_hash,
                                      added_suites=len(supplements) - start_count))
        manifest_input = digest(internal) + "".join(row["sha256"] for row in extra_sources)
        manifest = hashlib.sha256(manifest_input.encode("ascii")).hexdigest()
        library = RNAFragmentLibrary(metadata["suites"] + supplements, source_manifest_sha256=manifest,
                                     chemistry_registry_sha256=metadata["chemistry_registry_sha256"])
    structure = gemmi.read_structure(str(source))
    report = dict(target=entry, target_sha256=target_hash, internal_sha256=digest(internal),
                  terminal_sha256=digest(terminal), library_contexts=library.available_sequences,
                  path_limits=PATH_LIMITS, gap_limits=GAP_LIMITS, chains={})
    report["supplemental_internal_source"] = (
        dict(extra_sources[0], manifest_sha256=manifest) if len(extra_sources) == 1 else None
    )
    if len(extra_sources) > 1:
        report["supplemental_internal_sources"] = extra_sources
        report["combined_internal_manifest_sha256"] = manifest
    if sensitivity:
        report["sensitivity_scope"] = "Single-gate ablations; exploratory only, not acceptance."
        reference = REFERENCE_DIR.parents[1] / "gap_benchmarks/backbone_reference_audit.json"
        calibration = json.loads(reference.read_bytes())["sources"]["4U3L"]
        reference_cif = REFERENCE_DIR / "ion_series/4u3l.cif"
        if digest(reference_cif) != calibration["sha256"]:
            raise ValueError("4U3L reference coordinates do not match the audit")
        signs = [ribose_signs(_anchor_heavy_atoms(r), r.name)
                 for c in gemmi.read_structure(str(reference_cif))[0] for r in c
                 if r.name in {"A", "C", "G", "U"}]
        if not signs or None in signs[0].values() or any(s != signs[0] for s in signs):
            raise ValueError("4U3L does not give consistent reference chirality signs")
        target = np.median([[row[k] for k in FIELDS] for row in calibration["links"]], axis=0)
        report["rigid_gap_trial_reference"] = dict(
            entry="4U3L", audit_sha256=digest(reference), fields=FIELDS, medians=target.tolist(),
            max_shift_A=.5, max_phosphate_turn_deg=20., torsion_fields=TORSIONS,
            selection_policy="rigid_pass_then_phosphate_pass; torsions descriptive only")
    supported = set(library.available_sequences)
    for chain in structure[0]:
        if window_filter is not None and chain.name != window_filter[0]:
            continue
        residues = [r for r in chain if r.name in {"A", "C", "G", "U"}]
        if len(residues) < 3:
            continue
        seq = "".join(r.name for r in residues)
        if any(b.seqid.num != a.seqid.num + 1 for a, b in zip(residues, residues[1:])):
            raise ValueError("nonconsecutive target chain")
        if long_paths_only:
            windows = []
            for length in (4, 5, 6):
                for i in range(len(seq)-length+1):
                    segment = seq[i:i+length]
                    window = residues[i:i+length]
                    if window_filter is not None and (chain.name, str(window[0].seqid), length) != window_filter:
                        continue
                    missing = sorted({segment[j:j+2] for j in range(length-1)} - supported)
                    row = dict(start=str(window[0].seqid), sequence=segment, missing_contexts=missing)
                    if frame_preflight:
                        if not missing:
                            frames = np.stack([_make_frame(r) for r in window])
                            pairs, misses = screen_frame_pairs(library, segment, frames, PATH_LIMITS,
                                                               report_frame_misses=True)
                            counts = list(map(len, pairs))
                            rmsd_paths, motion_paths = count_overlap_paths(library, pairs, PATH_LIMITS)
                            row["frame_preflight"] = dict(
                                frame_candidates=counts, frame_misses=misses,
                                combinations_before_overlap=prod(counts),
                                overlap_rmsd_paths=rmsd_paths,
                                overlap_displacement_bound_paths=motion_paths)
                        windows.append(row)
                        continue
                    if not missing:
                        frames = np.stack([_make_frame(r) for r in window])
                        def audit_window(assembly):
                            return audit_assembled_context(
                                assembly, structure[0], chain.name, window, signs[0], target[0])

                        for key, reverse, joint, frame_bound in (("probe", False, False, False),
                                                     ("reverse_source_probe", True, False, False),
                                                     ("joint_probe", False, True, False),
                                                     ("joint_frame_probe", False, True, True)):
                            row[key] = probe_shared_residue(
                                library, segment, frames, PATH_LIMITS, make_base_frame,
                                audit_context=audit_window, reverse_source=reverse, joint=joint,
                                constrain_frames=frame_bound, report_frame_misses=key == "probe")
                        boundaries = []
                        for neighbor_index, key, atom in ((i-1, (0, "P"), "O3'"),
                                (i+length, (length-1, "O3'"), "P")):
                            if 0 <= neighbor_index < len(residues):
                                neighbors = [a for a in residues[neighbor_index] if a.name == atom
                                             and a.occ > 0.]
                                if len(neighbors) == 1:
                                    pos = neighbors[0].pos
                                    boundaries.append((key, np.array([pos.x, pos.y, pos.z]), target[0]))
                        expected = int(i > 0) + int(i+length < len(residues))
                        if boundaries and len(boundaries) == expected:
                            variants = [("joint_boundary_probe", False, False)]
                            downstream_angle = None
                            if row["probe"]["closest_rejected_overlap_by_step"]:
                                variants.extend((("displacement_bound_probe", True, False),
                                                 ("reverse_displacement_bound_probe", True, True)))
                                if i > 0:
                                    variants.extend((("upstream_phosphate_probe", True, True),
                                                     ("upstream_backbone_probe", True, True)))
                                    if i+length < len(residues):
                                        oxygens = [a.pos for a in residues[i+length]
                                                   if a.name == "O5'" and a.occ > 0.]
                                        if len(oxygens) == 1:
                                            key, phosphorus, _ = boundaries[-1]
                                            o5 = oxygens[0]
                                            downstream_angle = (key, phosphorus, np.array([o5.x, o5.y, o5.z]),
                                                                *ANGLE_WINDOWS["o3_p_o5_deg"])
                                            variants.extend((("upstream_backbone_angle_probe", True, True),
                                                             ("terminal_backbone_angle_probe", True, True)))
                            for key, use_bound, reverse in variants:
                                p = probe_shared_residue(
                                    library, segment, frames, PATH_LIMITS, make_base_frame,
                                    audit_context=audit_window, joint=True, constrain_frames=True,
                                    boundaries=tuple(boundaries), use_displacement_bound=use_bound,
                                    reverse_source=reverse,
                                    upstream_torsions={"upstream_phosphate_probe": 1,
                                                       "upstream_backbone_probe": 2,
                                                       "upstream_backbone_angle_probe": 2,
                                                       "terminal_backbone_angle_probe": 2}.get(key, 0),
                                    downstream_angle=(downstream_angle if key in ("upstream_backbone_angle_probe",
                                                                                 "terminal_backbone_angle_probe")
                                                      else None),
                                    rotate_downstream_o3=key == "terminal_backbone_angle_probe")
                                row[key] = p
                                if p["best"] is not None:
                                    context = p["best"]["context"]
                                    links = context["boundary_initial_geometry"]
                                    p["best"]["passed_context_without_second_adjustment"] = bool(
                                        len(links) == expected and not context["contacts"]
                                        and context["all_ribose_signs_match"]
                                        and all(abs(link["o3p_A"]-target[0]) <= .10 and
                                                all(link[k] is not None and lo <= link[k] <= hi
                                                    for k, (lo, hi) in ANGLE_WINDOWS.items())
                                                for link in (*links, *p["best"]["qc"]["links"])))
                                    if not p["best"]["passed_context_without_second_adjustment"]:
                                        p["best"].pop("candidate_geometry", None)
                    windows.append(row)
                    if progress:
                        terminal_probe = row.get("terminal_backbone_angle_probe", {})
                        print(f"{entry}:{chain.name}:{row['start']} {segment} "
                              f"terminal_paths={terminal_probe.get('paths_evaluated', 0)}",
                              file=sys.stderr, flush=True)
            if windows or window_filter is None:
                report["chains"][chain.name] = dict(sequence=seq, windows=windows)
            continue
        environment = _environment_heavy_atoms(structure, chain.name)
        windows = []
        for i in range(len(seq)-2):
            triplet = seq[i:i+3]
            missing = sorted({triplet[:2], triplet[1:]} - supported)
            row = dict(start=str(residues[i].seqid), sequence=triplet, missing_contexts=missing)
            if missing:
                row.update(path="unsupported_context", gap="unsupported_context")
                windows.append(row)
                continue
            start, middle, end = residues[i:i+3]
            def close(limits):
                return close_rna_gap_from_anchors(
                    library, triplet, _make_frame(start), _make_frame(end),
                    _anchor_heavy_atoms(start), _anchor_heavy_atoms(end), coordinate_unit="angstrom",
                    environment_atoms=environment, **limits)

            result = close(GAP_LIMITS)
            ablation = None
            if sensitivity:
                strict_pinned = close(dict(GAP_LIMITS, pin_anchor_coordinates=True))
                row["gap_strict_exact_anchors"] = strict_pinned.to_report()
                unique = close(dict(GAP_LIMITS, pin_anchor_coordinates=True,
                                    deduplicate_missing_geometry=True))
                row["gap_strict_unique_exact_anchors"] = unique.to_report()
                angle_screened = close(dict(GAP_LIMITS, pin_anchor_coordinates=True,
                                            deduplicate_missing_geometry=True,
                                            anchor_angle_windows_deg=ANGLE_WINDOWS))
                row["gap_strict_exact_anchors_with_angles"] = angle_screened.to_report()
                ablation = close(dict(GAP_LIMITS, max_anchor_atom_displacement_A=2.0))
                row["gap_sensitivity"] = ablation.to_report()
                pinned = close(dict(GAP_LIMITS, max_anchor_atom_displacement_A=2.0,
                                    pin_anchor_coordinates=True))
                row["gap_sensitivity_exact_anchors"] = pinned.to_report()
                row["gap_sensitivity_exact_anchor_qc"] = [
                    measure_open_chain_geometry(c.assembly.to_mdtraj()) for c in pinned.candidates]
            row["gap"] = result.to_report()
            truth = _anchor_heavy_atoms(middle)

            def score_candidates(closure):
                return [
                    _rmsd(truth, {name: xyz for (j, name), xyz in _xyz_by_residue(c.assembly).items()
                                  if j == 1}, set(truth)) for c in closure.candidates]

            row["gap_rmsd_A_and_atom_count"] = score_candidates(result)
            if ablation is not None:
                row["gap_strict_exact_anchors_rmsd_A_and_atom_count"] = score_candidates(strict_pinned)
                row["gap_strict_unique_exact_anchors_rmsd_A_and_atom_count"] = score_candidates(unique)
                row["gap_strict_exact_anchors_with_angles_rmsd_A_and_atom_count"] = score_candidates(angle_screened)
                row["gap_strict_exact_anchor_qc"] = [
                    measure_open_chain_geometry(c.assembly.to_mdtraj()) for c in strict_pinned.candidates]
                row["gap_sensitivity_exact_anchors_rmsd_A_and_atom_count"] = score_candidates(pinned)
                row["rigid_gap_trials"] = []
                row["phosphate_gap_trials"] = []
                for candidate in pinned.candidates:
                    original = {name: xyz for (j, name), xyz in _xyz_by_residue(candidate.assembly).items()
                                if j == 1}
                    for key, cap in (("rigid_gap_trials", 0.), ("phosphate_gap_trials", 20.)):
                        trial, fit = fit_missing_residue(
                            measure_link_coordinates, _anchor_heavy_atoms(start), original,
                            _anchor_heavy_atoms(end), target, max_phosphate_turn_deg=cap)
                        trajectory = candidate.assembly.to_mdtraj()
                        before_qc = measure_open_chain_geometry(trajectory)
                        for atom in trajectory.topology.residue(1).atoms:
                            trajectory.xyz[0, atom.index] = trial[atom.name]/10
                        after_qc = measure_open_chain_geometry(trajectory)
                        env_xyz = np.array([item["xyz_A"] for item in environment]).reshape(-1, 3)
                        env_min = (float(np.linalg.norm(
                            np.array(list(trial.values()))[:, None] - env_xyz, axis=-1).min())
                                   if len(env_xyz) else None)
                        eligible = (fit["converged"] and fit["after_in_window"]
                                    and fit["max_shift_A"] <= .5
                                    and fit["final_error"] < fit["initial_error"]
                                    and after_qc["short_heavy_contacts"] <= before_qc["short_heavy_contacts"]
                                    and (env_min is None or env_min >= GAP_LIMITS["min_environment_distance_A"]))
                        row[key].append(dict(
                            fit=fit, passed_diagnostic=eligible, after_qc=after_qc,
                            torsion_reference_rms_deg=torsion_diagnostic(after_qc, calibration["links"]),
                            environment_min_A=env_min,
                            rmsd_A_and_atom_count=_rmsd(truth, trial, set(truth))))
                row["selected_gap_trial_modes"] = [choose_trial(a, b) for a, b in
                    zip(row["rigid_gap_trials"], row["phosphate_gap_trials"], strict=True)]
                row["gap_sensitivity_rmsd_A_and_atom_count"] = score_candidates(ablation)
                row["gap_sensitivity_pinned_audit"] = []
                for candidate in ablation.candidates:
                    trajectory = candidate.assembly.to_mdtraj()
                    anchors = {0: _anchor_heavy_atoms(start), 2: _anchor_heavy_atoms(end)}
                    for residue in trajectory.topology.residues:
                        if residue.index in anchors:
                            for atom in residue.atoms:
                                if atom.name in anchors[residue.index]:
                                    trajectory.xyz[0, atom.index] = anchors[residue.index][atom.name] / 10
                    qc = measure_open_chain_geometry(trajectory)
                    row["gap_sensitivity_pinned_audit"].append(dict(
                        qc=qc, observed_links=[measure_link_coordinates(anchors[0], truth),
                                               measure_link_coordinates(truth, anchors[2])]))
            try:
                path = build_rna_chain_from_frames(
                    library, triplet, np.stack([_make_frame(r) for r in (start, middle, end)]),
                    coordinate_unit="angstrom", include_terminal_suites=False, **PATH_LIMITS)
            except (NoCompatibleRNAFragmentPathError, ValueError) as exc:
                row["path"] = dict(status="rejected", reason=str(exc))
            else:
                row["path"] = dict(status="assembled", selection=path.to_report(),
                                   qc=measure_open_chain_geometry(path.assembly.to_mdtraj()))
            if sensitivity:
                try:
                    diagnostics = {}
                    row["balanced_path_diagnostics"] = diagnostics
                    balanced = build_rna_chain_from_frames(
                        library, triplet, np.stack([_make_frame(r) for r in (start, middle, end)]),
                        coordinate_unit="angstrom", include_terminal_suites=False,
                        balance_frames=True, diagnostics=diagnostics, **PATH_LIMITS)
                except (NoCompatibleRNAFragmentPathError, ValueError) as exc:
                    row["balanced_path"] = dict(status="rejected", reason=str(exc))
                else:
                    row["balanced_path"] = dict(status="assembled", selection=balanced.to_report(),
                                                qc=measure_open_chain_geometry(balanced.assembly.to_mdtraj()),
                                                context=audit_assembled_context(
                                                    balanced.assembly, structure[0], chain.name,
                                                    (start, middle, end), signs[0], target[0]))
                if row["balanced_path"]["status"] == "rejected":
                    row["shared_residue_rigid_probe"] = probe_shared_residue(
                        library, triplet, np.stack([_make_frame(r) for r in (start, middle, end)]),
                        PATH_LIMITS, make_base_frame,
                        audit_context=lambda assembly: audit_assembled_context(
                            assembly, structure[0], chain.name, (start, middle, end), signs[0], target[0]))
                for field, value in (("max_translation_residual_A", 1.0),
                                     ("max_rotation_residual_degrees", 20.0)):
                    limits = dict(PATH_LIMITS, **{field: value})
                    try:
                        variant = build_rna_chain_from_frames(
                            library, triplet, np.stack([_make_frame(r) for r in (start, middle, end)]),
                            coordinate_unit="angstrom", include_terminal_suites=False, **limits)
                    except (NoCompatibleRNAFragmentPathError, ValueError) as exc:
                        result_row = dict(status="rejected", reason=str(exc))
                    else:
                        result_row = dict(status="assembled", selection=variant.to_report(),
                                          qc=measure_open_chain_geometry(variant.assembly.to_mdtraj()))
                    row.setdefault("path_sensitivity", {})[field] = dict(value=value, **result_row)
            windows.append(row)
        if supplement:
            report["chains"][chain.name] = dict(sequence=seq, full_path=dict(status="not_run_internal_only"),
                                                windows=windows)
            continue
        try:
            build_rna_chain_from_frames(
                library, seq, np.stack([_make_frame(r) for r in residues]), coordinate_unit="angstrom",
                include_terminal_suites=True, **PATH_LIMITS)
        except (UnsupportedSuiteSequenceError, NoCompatibleRNAFragmentPathError, ValueError) as exc:
            full = dict(status=type(exc).__name__, reason=str(exc))
        else:
            full = dict(status="assembled")
        report["chains"][chain.name] = dict(sequence=seq, full_path=full, windows=windows)
    if not report["chains"]:
        if window_filter is not None:
            raise ValueError(f"no eligible standard RNA window matched {window_filter!r}")
        raise ValueError("no standard RNA chains in target")
    if window_filter is not None:
        report["window_filter"] = dict(chain=window_filter[0], start=window_filter[1], length=window_filter[2])
    if frame_preflight:
        report["scope"] = "Template, frame, and pairwise overlap screening only; no fitting or assembly attempted."
        rows = [w for c in report["chains"].values() for w in c["windows"]]
        report["length_summary"] = {}
        for length in (4, 5, 6):
            selected = [w for w in rows if len(w["sequence"]) == length]
            screens = [w["frame_preflight"] for w in selected if "frame_preflight" in w]
            report["length_summary"][length] = dict(
                total=len(selected), covered=len(screens),
                all_steps_have_frame_candidates=sum(p["combinations_before_overlap"] > 0 for p in screens),
                combinations_before_overlap=sum(p["combinations_before_overlap"] for p in screens),
                overlap_rmsd_paths=sum(p["overlap_rmsd_paths"] for p in screens),
                overlap_displacement_bound_paths=sum(p["overlap_displacement_bound_paths"] for p in screens))
    elif not long_paths_only:
        report["window_summary"] = summarize_windows(report["chains"])
    else:
        rows = [w for c in report["chains"].values() for w in c["windows"]]
        report["length_summary"] = {}
        for length in (4, 5, 6):
            selected = [w for w in rows if len(w["sequence"]) == length]
            probes = [w["probe"] for w in selected if "probe" in w]
            trials = [((p["best"] or {}).get("context", {})
                       .get("boundary_frame_constrained_trial") or {}) for p in probes]
            joint = [w["joint_probe"] for w in selected if "joint_probe" in w]
            joint_frame = [w["joint_frame_probe"] for w in selected if "joint_frame_probe" in w]
            joint_boundary = [w["joint_boundary_probe"] for w in selected if "joint_boundary_probe" in w]
            terminal_backbone = [w["terminal_backbone_angle_probe"] for w in selected
                                 if "terminal_backbone_angle_probe" in w]
            joint_trials = [((p["best"] or {}).get("context", {})
                             .get("boundary_frame_constrained_trial") or {}) for p in joint]
            report["length_summary"][length] = dict(
                total=len(selected), covered=len(probes),
                missing_template_context=len(selected) - len(probes),
                missing_frame_candidate=sum(0 in p["frame_candidates"] for p in probes),
                baseline_paths_evaluated=sum(p["paths_evaluated"] for p in probes),
                joint_boundary_paths_evaluated=sum(p["paths_evaluated"] for p in joint_boundary),
                terminal_backbone_attempted_windows=len(terminal_backbone),
                terminal_backbone_paths_evaluated=sum(p["paths_evaluated"] for p in terminal_backbone),
                terminal_backbone_context_passed=sum(bool((p["best"] or {})
                    .get("passed_context_without_second_adjustment")) for p in terminal_backbone),
                assembled=sum(p["best"] is not None for p in probes),
                context_passed=sum(bool(t.get("passed_context_diagnostic")) for t in trials),
                joint_assembled=sum(p["best"] is not None for p in joint),
                joint_context_passed=sum(bool(t.get("passed_context_diagnostic")) for t in joint_trials),
                joint_frame_assembled=sum(p["best"] is not None for p in joint_frame),
                joint_boundary_assembled=sum(p["best"] is not None for p in joint_boundary),
                joint_boundary_context_passed=sum(bool((p["best"] or {})
                    .get("passed_context_without_second_adjustment")) for p in joint_boundary),
                joint_frame_context_passed=sum(bool((((p["best"] or {}).get("context") or {})
                    .get("boundary_frame_constrained_trial") or {}).get("passed_context_diagnostic"))
                    for p in joint_frame),
                search_truncated=sum(p["search_truncated"] for p in probes))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", choices=("1RNA", "1MIS", "157D", "420D", "438D", "2G3S", "4JRT", "8FEP"),
                        default="1RNA")
    parser.add_argument("--supplement", choices=("1RNA", "1RXB", "157D", "255D", "420D", "438D"),
                        action="append")
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--long-paths-only", action="store_true")
    parser.add_argument("--frame-preflight", action="store_true",
                        help="Screen 4–6-residue windows through the pairwise overlap checks without fitting.")
    parser.add_argument("--progress", action="store_true", help="Report each finished long-path window to stderr.")
    parser.add_argument("--window", nargs=3, metavar=("CHAIN", "START", "LENGTH"),
                        help="Write a separate report for one 4–6-residue window.")
    args = parser.parse_args()
    if args.frame_preflight and (args.long_paths_only or args.sensitivity or args.progress):
        parser.error("--frame-preflight is a separate report mode")
    if args.progress and not args.long_paths_only:
        parser.error("--progress requires --long-paths-only")
    window_filter = None
    if args.window:
        chain, start, length = args.window
        if not (args.long_paths_only or args.frame_preflight):
            parser.error("--window requires --long-paths-only or --frame-preflight")
        if not re.fullmatch(r"[A-Za-z0-9]+", chain) or not re.fullmatch(r"-?[0-9]+[A-Za-z]?", start):
            parser.error("--window expects an alphanumeric chain and an author residue number")
        if length not in {"4", "5", "6"}:
            parser.error("--window length must be 4, 5, or 6")
        window_filter = (chain, start, int(length))
    suffix = "".join(f"_plus_{entry.lower()}" for entry in (args.supplement or ()))
    if args.frame_preflight:
        suffix += "_frame_preflight"
    elif args.long_paths_only:
        suffix += "_long_paths"
    elif args.sensitivity:
        suffix += "_sensitivity"
    if window_filter is not None:
        suffix += f"_{chain}_{start}_{length}"
    output = REFERENCE_DIR.parent / f"gap_benchmarks/{args.entry.lower()}_transfer{suffix}.json"
    output.write_text(json.dumps(evaluate(args.entry, args.supplement,
                                          sensitivity=args.sensitivity or args.long_paths_only,
                                          long_paths_only=args.long_paths_only, progress=args.progress,
                                          frame_preflight=args.frame_preflight, window_filter=window_filter),
                                 indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
