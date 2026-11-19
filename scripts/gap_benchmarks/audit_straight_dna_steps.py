"""Audit all directed standard DNA steps in a straight duplex."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# Linearized order-2 de Bruijn cycle: every directed pair occurs once.
SEQUENCE = "AACAGATCCGCTGGTTA"
SCREEN_A = (1.30, 2.00)
CANONICAL = {f"D{b}": b for b in "ACGT"}


def summarize(report):
    """Join chains by 5'→3' step identity, not frame index."""
    steps = {}
    for link in report["links"]:
        name = "".join(CANONICAL[link[f"{side}_residue_name"]] for side in ("left", "right"))
        steps.setdefault(name, []).append(link)
    expected = {a + b for a in "ACGT" for b in "ACGT"}
    if steps.keys() != expected or any(
        sorted(row["chain_index"] for row in rows) != [0, 1] for rows in steps.values()
    ):
        raise ValueError("expected all 16 directed steps once on each strand")
    classifications = {}
    for step, rows in sorted(steps.items()):
        distances = [row["o3p_A"] for row in rows]
        if any(d is None for d in distances):
            raise ValueError(f"missing distance for {step}")
        classifications[step] = dict(
            distances_A=distances,
            strand_distance_delta_A=abs(distances[0] - distances[1]),
            both_outside_screen=all(not SCREEN_A[0] <= d <= SCREEN_A[1] for d in distances),
        )
    return classifications


def twist_scan(trajectory, frames):
    """Scan uniform twist in a common frame, keeping template atoms rigid."""
    residues = list(next(iter(trajectory.topology.chains)).residues)
    xyz = trajectory.xyz[0]
    p, o3 = [], []
    for res, frame in zip(residues, frames, strict=True):
        p.append(frame[1:] @ (xyz[res.atom("P").index] - frame[0]))
        o3.append(frame[1:] @ (xyz[res.atom("O3'").index] - frame[0]))
    p, o3 = np.array(p)[1:], np.array(o3)[:-1]
    tangents = frames[:-1, 3]
    rise = np.sum(np.diff(frames[:, 0], axis=0) * tangents, axis=1)
    observed_twist = np.degrees(np.arctan2(
        np.sum(frames[1:, 1] * frames[:-1, 2], axis=1),
        np.sum(frames[1:, 1] * frames[:-1, 1], axis=1),
    ))

    def distances(angles):
        t = np.radians(angles)
        c, s = np.cos(t), np.sin(t)
        x, y, z = p.T
        vectors = np.array([c*x - s*y, s*x + c*y, z + rise]).T - o3
        return np.linalg.norm(vectors, axis=1) * 10

    actual = [row["o3p_A"] for row in measure_links(trajectory)]
    checks = []
    low, high = SCREEN_A
    for angle in np.linspace(-180, 180, 1440, endpoint=False):
        d = distances(angle)
        outside = np.maximum(0., low - d) + np.maximum(0., d - high)
        checks.append((int(np.count_nonzero(outside)), float(np.max(outside)), angle))
    count, worst, angle = min(checks)
    return dict(grid_step_deg=0.25, observed_twist_range_deg=[float(observed_twist.min()),
                                                            float(observed_twist.max())],
                measured_reconstruction_max_delta_A=float(np.max(np.abs(distances(observed_twist) - actual))),
                best_sampled_twist_deg=float(angle), best_sampled_outside_steps=count,
                best_sampled_max_excess_A=worst)


def measure_links(trajectory):
    from mdna.structure_qc import measure_open_chain_geometry

    return [row for row in measure_open_chain_geometry(trajectory)["links"] if row["chain_index"] == 0]


def audit():
    sys.path.insert(0, str(ROOT))
    from mdna import make
    from mdna.structure_qc import measure_open_chain_geometry

    fraction = np.linspace(0., 1., 12)
    models = {}
    for name, axis in (("along_z", [0., 0., 1.]), ("oblique", [2., -3., 4.])):
        axis = np.asarray(axis)
        points = fraction[:, None] * axis[None, :] * (5.44 / np.linalg.norm(axis))
        model = make(sequence=SEQUENCE, control_points=points)
        trajectory = model.get_traj()
        report = measure_open_chain_geometry(trajectory)
        frames = model.get_frames()[:, 0, :, :]
        models[name] = dict(
            by_step=summarize(report), geometry=report,
            min_rise_nm=float(np.linalg.norm(np.diff(frames[:, 0], axis=0), axis=1).min()),
            max_rise_nm=float(np.linalg.norm(np.diff(frames[:, 0], axis=0), axis=1).max()),
        )
        if name == "along_z":
            models[name]["uniform_twist_scan"] = twist_scan(trajectory, frames)
    delta = [
        np.max(np.abs(np.subtract(row["distances_A"], models["oblique"]["by_step"][step]["distances_A"])))
        for step, row in models["along_z"]["by_step"].items()
    ]
    return dict(sequence=SEQUENCE, screen_A=SCREEN_A, models=models,
                max_orientation_distance_delta_A=float(max(delta)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        report = audit()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
