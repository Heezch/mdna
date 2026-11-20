"""Compare tangent and double-helix phase on identical standard DNA centers."""

import argparse
from contextlib import redirect_stdout
from hashlib import sha256
from itertools import zip_longest
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from evaluate_open_path_bends import ROOT, controls, screen_atomic_geometry
from probe_amber_anchor_restraints import _glycosidic_base_group
from probe_open_path_resampling import duplex_atom_frames, frames_on_grid
from probe_resampled_orientation import angle_outliers, rotate_frames


def fixed_heavy_indexes(topology):
    fixed = set()
    for chain in topology.chains():
        residues = list(chain.residues())
        for residue in residues:
            _, base_heavy = _glycosidic_base_group(topology, residue)
            fixed.update(base_heavy)
            fixed.update(atom.index for atom in residue.atoms() if atom.name == "C1'"
                         or residue in (residues[0], residues[-1]))
    return fixed


def describe(frames, trajectory, qc, fixed):
    atoms = list(trajectory.topology.atoms)
    by_atom = duplex_atom_frames(trajectory.topology)
    by_residue = {atom.residue.index: by_atom[atom.index] for atom in atoms}
    centers = frames[:, 0]

    def placement(i, j):
        return dict(frames=[i, j], frame_span=abs(i - j),
                    center_distance_A=float(np.linalg.norm(centers[i] - centers[j]) * 10.),
                    tangent_change_deg=float(np.degrees(np.arccos(np.clip(frames[i, 3] @ frames[j, 3], -1., 1.)))))

    def label(index):
        atom = atoms[index]
        return [atom.residue.index, atom.residue.name, atom.name]

    pairs = [(i, j, d) for i, j, d in qc["short_heavy_contact_pairs"] if i in fixed and j in fixed]
    close_pairs = sorted(pairs, key=lambda pair: pair[2])[:5]
    longest = sorted((row for row in qc["links"] if row["o3p_A"] is not None),
                     key=lambda row: -row["o3p_A"])[:5]
    return dict(fixed_contacts=len(pairs), fixed_contacts_below_1A=sum(d < 1. for _, _, d in pairs),
                short_heavy_contacts=qc["short_heavy_contacts"],
                closest_fixed_pairs=[dict(atom_i=label(i), atom_j=label(j), distance_A=d,
                                          **placement(by_atom[i], by_atom[j])) for i, j, d in close_pairs],
                longest_links=[dict(**row, **placement(by_residue[row["left_residue_index"]],
                                                       by_residue[row["right_residue_index"]])) for row in longest],
                links_above_5A=sum(row["o3p_A"] is not None and row["o3p_A"] > 5. for row in qc["links"]),
                angle_outliers=angle_outliers(qc["links"]), triage=screen_atomic_geometry(qc))


def evaluate(source, *, joint_phase=False):
    from mdna.generators import StructureGenerator
    from mdna.structure_qc import measure_open_chain_geometry

    raw = source.read_bytes()
    previous = json.loads(raw)
    frozen_raw = source.with_name(previous["source_report"]).read_bytes()
    if sha256(frozen_raw).hexdigest() != previous["source_sha256"]:
        raise ValueError("Frozen center report hash differs")
    frozen = json.loads(frozen_raw)
    seed, sequence = frozen["seed"], frozen["sequence"]
    centers = np.asarray(frozen["final"]["frame_origins_nm"])
    points = controls(seed)

    fixed = None

    def measure(frames):
        nonlocal fixed
        if not np.array_equal(frames[:, 0], centers):
            raise ValueError("Frame centers changed")
        traj = StructureGenerator(sequence=sequence, frames=frames).get_traj(remove_terminal_phosphates=True)
        traj = traj.atom_slice([atom.index for atom in traj.topology.atoms if atom.element.atomic_number > 1])
        if fixed is None:
            fixed = fixed_heavy_indexes(traj.topology.to_openmm())
        return describe(frames, traj, measure_open_chain_geometry(traj, include_contact_pairs=True), fixed)

    outgoing = frames_on_grid(points, centers)
    sweep, = [row for row in previous["sweeps"] if row["torsion_cap_deg"] == 60.]
    resumed = outgoing
    for step in sweep["steps"]:
        if step["status"] == "accepted":
            resumed = rotate_frames(resumed, step["frames"], step["local_basis_axis"], step["angles_deg"])
    baseline = measure(resumed)
    if baseline["short_heavy_contacts"] != sweep["final_short_heavy_contacts"]:
        raise ValueError("Frozen reference contact count differs after replay")
    report = dict(seed=seed, source_report=source.name, source_sha256=sha256(raw).hexdigest(),
                  fixed_frame_centers=True, baseline_60deg=baseline)
    if joint_phase:
        return dict(report, **refine_phase(resumed, baseline, measure, seed=seed))

    candidates = []
    for mode in ("outgoing", "bisector"):
        frames = frames_on_grid(points, centers, tangent_mode=mode)
        for phase in range(0, 360, 45):
            trial = frames.copy()
            rotations = Rotation.from_rotvec(trial[:, 3] * np.radians(phase))
            for i, rotation in enumerate(rotations):
                trial[i, 1:] = rotation.apply(trial[i, 1:])
            candidates.append(dict(tangent_mode=mode, phase_deg=phase, geometry=measure(trial)))

    def improves(row):
        stats = row["geometry"]
        return (stats["fixed_contacts"] < baseline["fixed_contacts"]
                and stats["links_above_5A"] < baseline["links_above_5A"]
                and stats["short_heavy_contacts"] <= baseline["short_heavy_contacts"])

    return dict(report, candidates=candidates,
                improved_candidates=[dict(tangent_mode=row["tangent_mode"], phase_deg=row["phase_deg"])
                                     for row in candidates if improves(row)])


def refine_phase(initial, baseline, measure, *, seed):
    """Try bounded local spin at frames shared by severe links or close contacts."""
    centers = initial[:, 0]
    offsets = np.zeros(len(initial), dtype=int)
    current, geometry = initial, baseline
    iterations = []
    keys = ("fixed_contacts_below_1A", "links_above_5A", "fixed_contacts", "short_heavy_contacts")

    def score(stats):
        return (*[stats[name] for name in keys], stats["longest_links"][0]["o3p_A"])

    termination = "round_limit"
    for round_index in range(1, 7):
        link_frames = [frame for link in geometry["longest_links"] if link["o3p_A"] > 5.
                       for frame in link["frames"]]
        contact_frames = [frame for pair in geometry["closest_fixed_pairs"] for frame in pair["frames"]]
        targets = list(dict.fromkeys(frame for pair in zip_longest(link_frames, contact_frames)
                                     for frame in pair if frame is not None and 0 < frame < len(initial) - 1))[:10]
        original_score = best_score = score(geometry)
        best = None
        row = dict(targeted_frames=targets, evaluated=0, preserves_screens=0)
        iterations.append(row)
        for frame in targets:
            for change in (-15, -10, -5, 5, 10, 15):
                requested = offsets[frame] + change
                if abs(requested) > 30 or np.any(abs(offsets[[frame - 1, frame + 1]] - requested) > 15):
                    continue
                trial = rotate_frames(current, [frame], 2, [change])
                stats = measure(trial)
                row["evaluated"] += 1
                if any(stats[name] > geometry[name] for name in keys):
                    continue
                row["preserves_screens"] += 1
                rank = score(stats)
                if rank[-1] <= original_score[-1] + 1e-8 and rank < best_score:
                    best_score = rank
                    best = frame, change, trial, stats
        if best is None:
            termination = "no_improving_move"
            print(f"{seed} phase round {round_index}: evaluated {row['evaluated']}; stopped", file=sys.stderr)
            break
        frame, change, current, geometry = best
        offsets[frame] += change
        row.update(accepted=dict(frame=frame, change_deg=change), score=best_score)
        print(f"{seed} phase round {round_index}: evaluated {row['evaluated']}; accepted frame {frame}",
              file=sys.stderr)

    if not np.array_equal(current[:, 0], centers):
        raise RuntimeError("Joint phase search moved frame centers")
    return dict(mode="joint_local_phase", termination=termination, rounds=iterations,
                cumulative_phase_deg=offsets.tolist(), final=geometry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--joint-phase", action="store_true",
                        help="Refine local axial phase using both long links and fixed-atom contacts.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    with redirect_stdout(sys.stderr):
        cases = [evaluate(path, joint_phase=args.joint_phase) for path in args.source]
    args.output.write_text(json.dumps(cases, indent=2, allow_nan=False) + "\n", encoding="utf-8")
