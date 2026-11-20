"""Replay fixed duplex centers and test local orientation at a long O3′–P link."""

import argparse
from contextlib import redirect_stdout
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from evaluate_open_path_bends import (
    LINK_ANGLE_WINDOWS_DEG as WINDOWS, LINK_TORSION_FIELDS as TORSIONS,
    O3P_SCREEN_A, ROOT, controls, screen_atomic_geometry,
)
from probe_open_path_resampling import frames_on_grid, new_link_outliers


def orientations(frames, target_frames):
    for axis in range(3):
        for degrees in (-15, -10, -5, 5, 10, 15):
            for directions in ((1, 0), (0, 1), (1, 1), (1, -1)):
                angles = [degrees * direction for direction in directions]
                yield axis, angles, rotate_frames(frames, target_frames, axis, angles)


def rotate_frames(frames, target_frames, axis, angles):
    trial = frames.copy()
    for frame, angle in zip(target_frames, angles, strict=True):
        rotation = Rotation.from_rotvec(frames[frame, 1 + axis] * np.radians(angle))
        trial[frame, 1:] = rotation.apply(frames[frame, 1:])
    return trial


def angle_outliers(links):
    return {name: sum(row[name] is None or not low <= row[name] <= high for row in links)
            for name, (low, high) in WINDOWS.items()}


def torsion_changes(before, after):
    summary = {}
    for name in TORSIONS:
        pairs = [(periodic_change(a[name], b[name]), a, b) for a, b in zip(before, after, strict=True)
                 if a[name] is not None and b[name] is not None]
        changes = [change for change, _, _ in pairs]
        summary[name] = dict(measured_pairs=len(changes),
                             p95_abs_change_deg=float(np.percentile(changes, 95)) if changes else None,
                             max_abs_change_deg=max(changes, default=None))
        if pairs:
            _, a, b = max(pairs, key=lambda pair: pair[0])
            summary[name]["max_change_link"] = dict(
                chain_index=a["chain_index"], left_residue_index=a["left_residue_index"],
                right_residue_index=a["right_residue_index"], before_deg=a[name], after_deg=b[name])
    return summary


def periodic_change(before, after):
    return abs((after - before + 180) % 360 - 180)


def evaluate(source, chain=None, target_frames=None, *, torsion_caps=()):
    sys.path.insert(0, str(ROOT))
    from mdna.nucleic import Nucleic
    from mdna.structure_qc import measure_open_chain_geometry

    raw = source.read_bytes()
    frozen = json.loads(raw)
    placement = frozen["final"]
    centers = np.asarray(placement["frame_origins_nm"])
    frames = frames_on_grid(controls(frozen["seed"]), centers)
    sequence = frozen["sequence"]
    n_frames = len(frames)
    def measure(oriented):
        model = Nucleic(sequence=sequence, n_bp=n_frames, frames=oriented, circular=False)
        if not np.array_equal(model.get_frames()[:, 0, 0, :], centers):
            raise RuntimeError("orientation search moved a fixed frame center")
        return measure_open_chain_geometry(model.get_traj())

    base_qc = measure(frames)
    base_triage = screen_atomic_geometry(base_qc)
    base_triage["o3p_screen_A"] = list(base_triage["o3p_screen_A"])
    if (base_qc["short_heavy_contacts"] != placement["result"]["short_heavy_contacts"]
            or base_triage != placement["result"]["triage"]):
        raise ValueError("replayed structure geometry differs from the frozen input")
    report = dict(seed=frozen["seed"], source_sha256=sha256(raw).hexdigest(), source_report=source.name,
                  fixed_frame_centers=True, baseline_short_heavy_contacts=base_qc["short_heavy_contacts"],
                  baseline_triage=base_triage)
    if torsion_caps:
        measured = {}

        def measure_once(oriented):
            key = oriented[:, 1:].tobytes()
            if key not in measured:
                measured[key] = measure(oriented)
            return measured[key]

        return dict(report, mode="torsion_cap_comparison", sweeps=[
            sweep_orientations(frames, base_qc, frozen["final_new_outliers"], measure_once, torsion_cap=cap)
            for cap in (None, *torsion_caps)])
    if target_frames is None:
        return dict(report, **sweep_orientations(frames, base_qc, frozen["final_new_outliers"], measure))

    if (abs(target_frames[0] - target_frames[1]) != 1
            or not all(0 < frame < n_frames - 1 for frame in target_frames)):
        raise ValueError("target frames must be adjacent interior frames")
    chain_index = target_frames[0] if chain == 0 else n_frames - 1 - target_frames[0]
    target_index = chain * (n_frames - 1) + chain_index
    if (target_frames[1] - target_frames[0]) != (1 if chain == 0 else -1):
        raise ValueError("target frames must follow the selected chain's 5′ to 3′ direction")
    base_target = base_qc["links"][target_index]
    key = (base_target["chain_index"], base_target["left_residue_index"], base_target["right_residue_index"])
    frozen_link, = [row for row in frozen["final_new_outliers"] if row["chain_index"] == chain
                    and row["frames"] == list(target_frames)]
    if abs(base_target["o3p_A"] - frozen_link["observed_A"]) > 1e-8:
        raise ValueError("replayed target distance differs from the frozen input")

    low, high = O3P_SCREEN_A
    trials = []
    for axis, angles, trial in orientations(frames, target_frames):
        qc = measure(trial)
        target = qc["links"][target_index]
        others = [row for row in new_link_outliers(qc["links"], base_qc["links"], n_frames)
                  if (row["chain_index"], row["left_residue_index"], row["right_residue_index"]) != key]
        failures = []
        if not low <= target["o3p_A"] <= high:
            failures.append("target_distance")
        if others:
            failures.append("new_outliers_elsewhere")
        if qc["short_heavy_contacts"] > base_qc["short_heavy_contacts"]:
            failures.append("short_heavy_contacts")
        trials.append(dict(local_basis_axis=axis, angles_deg=angles, target=target,
                           new_outliers_elsewhere=others, short_heavy_contacts=qc["short_heavy_contacts"],
                           triage=screen_atomic_geometry(qc), failures=failures))

    selected = min((row for row in trials if not row["failures"]), default=None,
                   key=lambda row: (sum(abs(angle) for angle in row["angles_deg"]),
                                    row["short_heavy_contacts"], row["target"]["o3p_A"]))
    return dict(report, target_frames=target_frames, baseline_target=base_target, trials=trials, selected=selected)


def sweep_orientations(initial, base_qc, outliers, measure, *, torsion_cap=None):
    """Visit the frozen newly long links once, preserving whole-duplex screens."""
    low, high = O3P_SCREEN_A
    n_frames = len(initial)
    current, qc = initial, base_qc
    history = []
    for row in outliers:
        pair = row["frames"]
        if row["observed_A"] <= high or min(pair) == 0 or max(pair) == n_frames - 1:
            continue
        target = next(i for i, link in enumerate(qc["links"])
                      if link["chain_index"] == row["chain_index"]
                      and link["left_residue_index"] == row["left_residue_index"])
        distance = qc["links"][target]["o3p_A"]
        step = dict(chain_index=row["chain_index"], frames=pair, before_A=distance,
                    within_rotation_cap=0, distance_pass=0, contacts_and_links_pass=0, angle_pass=0,
                    torsion_pass=0)
        history.append(step)
        if low <= distance <= high:
            step["status"] = "already_in_window"
            continue
        old_angles = angle_outliers(qc["links"])
        best = None
        for axis, angles, trial in orientations(current, pair):
            delta = trial[pair, 1:] @ np.swapaxes(initial[pair, 1:], 1, 2)
            cumulative = np.degrees(Rotation.from_matrix(delta).magnitude())
            if np.any(cumulative > 15. + 1e-8):
                continue
            step["within_rotation_cap"] += 1
            observed = measure(trial)
            new_distance = observed["links"][target]["o3p_A"]
            if not low <= new_distance <= high:
                continue
            step["distance_pass"] += 1
            if (observed["short_heavy_contacts"] > qc["short_heavy_contacts"]
                    or new_link_outliers(observed["links"], qc["links"], n_frames)):
                continue
            step["contacts_and_links_pass"] += 1
            new_angles = angle_outliers(observed["links"])
            if any(new_angles[name] > old_angles[name] for name in WINDOWS):
                continue
            step["angle_pass"] += 1
            if torsion_cap is not None and any(
                b[name] is None or periodic_change(a[name], b[name]) > torsion_cap + 1e-8
                for a, b in zip(base_qc["links"], observed["links"], strict=True)
                for name in TORSIONS if a[name] is not None
            ):
                continue
            step["torsion_pass"] += 1
            rank = (sum(new_angles.values()), observed["short_heavy_contacts"], sum(abs(a) for a in angles))
            if best is None or rank < best[0]:
                best = rank, trial, observed, axis, angles, new_distance
        if best is None:
            step["status"] = "no_passing_orientation"
            continue
        _, current, qc, axis, angles, new_distance = best
        step.update(status="accepted", local_basis_axis=axis, angles_deg=angles, after_A=new_distance,
                    short_heavy_contacts=qc["short_heavy_contacts"], angle_outliers=angle_outliers(qc["links"]))

    delta = current[:, 1:] @ np.swapaxes(initial[:, 1:], 1, 2)
    return dict(mode="multi_link_sweep", angle_windows_deg=WINDOWS, torsion_cap_deg=torsion_cap,
                baseline_angle_outliers=angle_outliers(base_qc["links"]), steps=history,
                max_frame_rotation_deg=float(np.degrees(Rotation.from_matrix(delta).magnitude()).max()),
                final_angle_outliers=angle_outliers(qc["links"]), final_triage=screen_atomic_geometry(qc),
                final_short_heavy_contacts=qc["short_heavy_contacts"],
                torsion_changes=torsion_changes(base_qc["links"], qc["links"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--chain", type=int, choices=(0, 1))
    parser.add_argument("--frames", type=int, nargs=2)
    parser.add_argument("--sweep", action="store_true", help="Visit the frozen newly long links in order.")
    parser.add_argument("--torsion-cap-deg", type=float, nargs="+", default=(),
                        help="Compare sweep results with the supplied periodic torsion-change caps.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.sweep and (args.chain is not None or args.frames is not None)
            or not args.sweep and (args.chain is None or args.frames is None)):
        parser.error("select --sweep or provide both --chain and --frames")
    if args.torsion_cap_deg and (not args.sweep or any(not 0. <= cap <= 180. for cap in args.torsion_cap_deg)):
        parser.error("--torsion-cap-deg requires --sweep and finite values from 0 to 180 degrees")
    with redirect_stdout(sys.stderr):
        result = evaluate(args.source, args.chain, args.frames, torsion_caps=args.torsion_cap_deg)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
