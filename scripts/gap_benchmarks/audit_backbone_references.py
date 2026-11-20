"""Measure local experimental DNA/RNA geometry using the model QC metric."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy

import gemmi
import numpy as np

from evaluate_open_path_bends import LINK_ANGLE_WINDOWS_DEG as WINDOWS, LINK_TORSION_FIELDS as TORSIONS

ROOT = Path(__file__).resolve().parents[2]
MEASURE = runpy.run_path(str(ROOT / "mdna/structure_qc.py"))["measure_link_coordinates"]
SOURCES = {
    "119D": ("DNA", "artifacts/gap_benchmarks/119d_source.pdb"),
    "1WQY": ("DNA", "artifacts/gap_benchmarks/1wqy_source.pdb"),
    "4U3L": ("RNA", "artifacts/rna_pilot/reference/ion_series/4u3l.cif"),
}
REQUIRED = {"C3'", "C4'", "C5'", "O3'", "O5'", "P"}


def atom_map(residue):
    result = {}
    for atom in residue:
        if atom.name not in REQUIRED:
            continue
        if atom.altloc not in ("\x00", " ", "") or atom.name in result or atom.occ <= 0.:
            return None
        result[atom.name] = (atom.pos.x, atom.pos.y, atom.pos.z)
    return result


def read_source(chemistry, path):
    names = set("ACGU") if chemistry == "RNA" else {"DA", "DC", "DG", "DT"}
    rows, exclusions = [], []
    for chain in gemmi.read_structure(str(path))[0]:
        residues = list(chain)
        for left, right in zip(residues, residues[1:]):
            if left.name not in names or right.name not in names:
                continue
            row = dict(chain=chain.name, left=str(left.seqid), right=str(right.seqid),
                       step=left.name + "-" + right.name)
            if right.seqid.num != left.seqid.num+1 or str(left.seqid.icode).strip() or str(right.seqid.icode).strip():
                exclusions.append(dict(**row, reason="nonconsecutive_author_numbering"))
                continue
            a, b = atom_map(left), atom_map(right)
            if a is None or b is None:
                exclusions.append(dict(**row, reason="ambiguous_backbone_coordinates"))
                continue
            values = MEASURE(a, b)
            if any(v is None for v in values.values()):
                exclusions.append(dict(**row, reason="missing_or_degenerate_geometry"))
                continue
            rows.append(dict(**row, **values))
    return rows, exclusions


def summary(rows):
    return {
        name: dict(outside=sum(not low <= row[name] <= high for row in rows),
                   range_deg=[min(row[name] for row in rows), max(row[name] for row in rows)])
        for name, (low, high) in WINDOWS.items()
    }


def compare_torsions(reference, cases):
    """Nearest joint torsion RMS distance, treating each angle as periodic."""
    anchors = np.array([[row[k] for k in TORSIONS] for row in reference])
    result = {}
    for name, case in cases.items():
        result[name] = {}
        for stage in ("before", "after"):
            rows = case[stage]["links"]
            valid = [row for row in rows if all(row[k] is not None for k in TORSIONS)]
            probe = np.array([[row[k] for k in TORSIONS] for row in valid]).reshape(-1, len(TORSIONS))
            delta = (probe[:, None] - anchors + 180.) % 360. - 180.
            nearest = np.sqrt(np.mean(delta**2, axis=-1)).min(axis=1)
            result[name][stage] = dict(measured=len(valid), excluded=len(rows)-len(valid),
                                       median_deg=float(np.median(nearest)) if len(valid) else None,
                                       p90_deg=float(np.percentile(nearest, 90)) if len(valid) else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--probe", type=Path, help="Synthetic DNA probe JSON with torsion fields")
    args = parser.parse_args()
    report = dict(scope="Local experimental controls; not chemistry-wide cutoffs",
                  probe_angle_windows_deg=WINDOWS, sources={})
    for entry, (chemistry, relative) in SOURCES.items():
        path = ROOT / relative
        rows, excluded = read_source(chemistry, path)
        report["sources"][entry] = dict(chemistry=chemistry, path=relative,
                                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                        links=rows, excluded=excluded, angle_comparison=summary(rows),
                                        o3p_range_A=[min(r["o3p_A"] for r in rows),
                                                    max(r["o3p_A"] for r in rows)])
    if args.probe:
        probe_bytes = args.probe.read_bytes()
        dna = [row for source in report["sources"].values() if source["chemistry"] == "DNA"
               for row in source["links"]]
        report["synthetic_comparison"] = dict(
            path=str(args.probe), sha256=hashlib.sha256(probe_bytes).hexdigest(),
            reference_links=len(dna), torsions=TORSIONS,
            scope="Pooled DNA links, no sequence matching; this is a descriptive distance, not a gate",
            cases=compare_torsions(dna, json.loads(probe_bytes)["cases"]))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
