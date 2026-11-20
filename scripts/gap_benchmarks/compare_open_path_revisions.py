"""Compare two committed MDNA snapshots with identical DNA inputs and QC."""

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from evaluate_open_path_bends import BUDGET_NM, ROOT, SEED, SEQUENCE, controls, screen_atomic_geometry


def build_cases(source, *, regularized):
    sys.path.insert(0, str(source))
    import mdna

    if not Path(mdna.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError("MDNA was not imported from the selected revision.")
    qc = runpy.run_path(str(ROOT / "mdna/structure_qc.py"))["measure_open_chain_geometry"]
    turns = runpy.run_path(str(ROOT / "mdna/path_geometry.py"))["_turns"]
    points = controls()
    cases = [("straight", np.linspace(points[0], points[-1], len(points)), {}),
             ("original", points, {})]
    if regularized:
        cases.append(("regularized", points, {"path_deviation_nm": BUDGET_NM}))
    results = {}
    for key, coordinates, options in cases:
        model = mdna.make(sequence=SEQUENCE, control_points=coordinates, **options)
        trajectory = model.get_traj()
        origins = model.get_frames()[:, 0, 0, :]
        metrics = qc(trajectory)
        results[key] = dict(
            max_frame_turn_deg=float(turns(origins).max()),
            geometry=metrics, triage=screen_atomic_geometry(metrics),
            atom_labels=[(a.residue.chain.index, a.residue.index, a.residue.name, a.name)
                         for a in trajectory.topology.atoms],
            xyz_nm=trajectory.xyz[0].tolist(), frame_origins_nm=origins.tolist(),
        )
    return results


def run_revision(revision, *, regularized):
    commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"], cwd=ROOT, text=True).strip()
    data = subprocess.check_output(["git", "archive", "--format=zip", commit], cwd=ROOT)
    with TemporaryDirectory(prefix="mdna-revision-") as directory:
        with ZipFile(io.BytesIO(data)) as archive:
            archive.extractall(directory)
        args = [sys.executable, "-B", __file__, "--worker-source", directory]
        if regularized:
            args.append("--regularized")
        result = json.loads(subprocess.check_output(args, cwd=directory))
    return commit, result


def coordinate_comparison(first, second):
    if first["atom_labels"] != second["atom_labels"]:
        return {"same_atom_inventory_and_order": False}
    displacements = np.linalg.norm(np.asarray(second["xyz_nm"])-first["xyz_nm"], axis=1)
    frame_shift = np.linalg.norm(np.asarray(second["frame_origins_nm"])-first["frame_origins_nm"], axis=1)
    return dict(same_atom_inventory_and_order=True,
                atom_rms_displacement_nm=float(np.sqrt(np.mean(displacements**2))),
                max_atom_displacement_nm=float(displacements.max()),
                max_frame_shift_nm=float(frame_shift.max()),
                endpoint_shift_nm=frame_shift[[0, -1]].tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", default="8fa36bfd9876c23ac76372b2a5b3a7900cc40579")
    parser.add_argument("--fork", default="afc749496f5eee08c9bf1d7eb3cd1428a37a9c87")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--regularized", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_source is not None:
        with redirect_stdout(sys.stderr):
            result = build_cases(args.worker_source, regularized=args.regularized)
        print(json.dumps(result, allow_nan=False))
        return
    if args.output is None:
        parser.error("--output is required")
    old_commit, old = run_revision(args.upstream, regularized=False)
    new_commit, new = run_revision(args.fork, regularized=True)
    comparisons = {f"{case}_legacy_interface": coordinate_comparison(old[case], new[case])
                   for case in ("straight", "original")}
    comparisons["upstream_original_to_fork_regularized"] = coordinate_comparison(
        old["original"], new["regularized"])
    for results in (old, new):
        for case in results.values():
            for key in ("atom_labels", "xyz_nm", "frame_origins_nm"):
                del case[key]
    report = dict(seed=SEED, sequence=SEQUENCE, inputs_nm=controls().tolist(),
                  regularization_budget_nm=BUDGET_NM, upstream_commit=old_commit, fork_commit=new_commit,
                  comparisons=comparisons, upstream=old, fork=new)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
