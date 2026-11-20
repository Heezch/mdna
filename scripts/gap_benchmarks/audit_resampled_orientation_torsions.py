"""Compare replayed, fixed-center DNA models to local experimental torsions."""

import argparse
from contextlib import redirect_stdout
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

from evaluate_open_path_bends import LINK_TORSION_FIELDS as TORSIONS, ROOT, controls, screen_atomic_geometry
from probe_open_path_resampling import frames_on_grid
from probe_resampled_orientation import periodic_change, rotate_frames


def references(path):
    data = json.loads(path.read_bytes())
    panels = {}
    for entry, source in data["sources"].items():
        if source["chemistry"] != "DNA":
            continue
        if sha256((ROOT / source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"Reference coordinate hash changed for {entry}")
        by_step = {}
        for link in source["links"]:
            by_step.setdefault(link["step"], []).append([link[name] for name in TORSIONS])
        panels[entry] = dict(source_sha256=source["sha256"],
                             by_step={step: np.asarray(values) for step, values in by_step.items()})
    return panels


def compare(before, after, by_step, affected):
    rows = []
    for i, (old, new) in enumerate(zip(before, after, strict=True)):
        step = f'{old["left_residue_name"]}-{old["right_residue_name"]}'
        anchors = by_step.get(step)
        values = [[row[name] for name in TORSIONS] for row in (old, new)]
        if anchors is None or any(value is None for group in values for value in group):
            continue
        difference = (np.asarray(values)[:, None, :] - anchors + 180.) % 360. - 180.
        prior, updated = np.sqrt(np.mean(difference**2, axis=2)).min(axis=1)
        rows.append(dict(chain_index=old["chain_index"], left_residue_index=old["left_residue_index"],
                         right_residue_index=old["right_residue_index"], step=step, affected=i in affected,
                         reference_links=len(anchors), before_deg=float(prior), after_deg=float(updated)))

    def summary(selected, total):
        if not selected:
            return dict(scored=0, unscored=total)
        before_values = [row["before_deg"] for row in selected]
        after_values = [row["after_deg"] for row in selected]
        changes = np.asarray(after_values) - before_values
        return dict(scored=len(selected), unscored=total - len(selected),
                    median_deg=[float(np.median(before_values)), float(np.median(after_values))],
                    p90_deg=[float(np.percentile(before_values, 90)), float(np.percentile(after_values, 90))],
                    median_paired_change_deg=float(np.median(changes)),
                    improved=int(np.count_nonzero(changes < -.01)), worsened=int(np.count_nonzero(changes > .01)))

    return dict(all_links=summary(rows, len(before)),
                affected_links=summary([row for row in rows if row["affected"]], len(affected)), links=rows)


def audit(path, panels):
    from mdna.nucleic import Nucleic
    from mdna.structure_qc import measure_open_chain_geometry

    raw = path.read_bytes()
    report = json.loads(raw)
    centers_raw = path.with_name(report["source_report"]).read_bytes()
    if sha256(centers_raw).hexdigest() != report["source_sha256"]:
        raise ValueError("Frozen center report hash differs")
    frozen = json.loads(centers_raw)
    centers = np.asarray(frozen["final"]["frame_origins_nm"])
    initial = frames_on_grid(controls(frozen["seed"]), centers)

    def measure(frames, triage, contacts):
        model = Nucleic(sequence=frozen["sequence"], n_bp=len(frames), frames=frames, circular=False)
        qc = measure_open_chain_geometry(model.get_traj())
        if (not np.array_equal(model.get_frames()[:, 0, 0, :], centers)
                or qc["short_heavy_contacts"] != contacts
                or json.loads(json.dumps(screen_atomic_geometry(qc))) != triage):
            raise ValueError("Fixed-center replay differs from reported geometry")
        return qc["links"]

    prior = measure(initial, report["baseline_triage"], report["baseline_short_heavy_contacts"])
    variants, affected = [], set()
    for sweep in report["sweeps"]:
        frames = initial
        for step in sweep["steps"]:
            if step["status"] == "accepted":
                frames = rotate_frames(frames, step["frames"], step["local_basis_axis"], step["angles_deg"])
        links = measure(frames, sweep["final_triage"], sweep["final_short_heavy_contacts"])
        affected.update(i for i, (old, new) in enumerate(zip(prior, links, strict=True))
                        if any(periodic_change(old[name], new[name]) > .01 for name in TORSIONS
                               if old[name] is not None and new[name] is not None))
        variants.append((sweep["torsion_cap_deg"], links))

    return dict(seed=frozen["seed"], source_report=path.name, source_sha256=sha256(raw).hexdigest(),
                affected_link_count=len(affected), sweeps=[
                    dict(torsion_cap_deg=cap, references={
                        entry: compare(prior, links, source["by_step"], affected)
                        for entry, source in panels.items()}) for cap, links in variants])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    panels = references(args.reference)
    with redirect_stdout(sys.stderr):
        cases = [audit(path, panels) for path in args.source]
    report = dict(scope="Same directed dinucleotide, nearest joint periodic RMS; descriptive comparison",
                  torsion_fields=TORSIONS, reference_sha256=sha256(args.reference.read_bytes()).hexdigest(),
                  reference_panels={entry: dict(source_sha256=source["source_sha256"],
                                                step_counts={step: len(values) for step, values in source["by_step"].items()})
                                    for entry, source in panels.items()}, cases=cases)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
