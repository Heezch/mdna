"""Compare deposited and rebuilt fragment contacts to crystal symmetry mates."""

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

import gemmi
import numpy as np


def crystal_contacts(coordinates, window_indices, cell, operations, cutoff):
    """Count directed window-to-image pairs strictly below the cutoff."""
    orth = np.array(cell.orth.mat.tolist())
    frac = np.array(cell.frac.mat.tolist())
    fractional = coordinates @ frac.T
    query = fractional[window_indices]
    # A Cartesian cutoff bounds each fractional component by the row norm.
    padding = cutoff * np.linalg.norm(frac, axis=1)
    contacts = []
    for op in operations:
        image = np.array([op.apply_to_xyz(point) for point in fractional])
        low = np.ceil(query.min(axis=0) - image.max(axis=0) - padding).astype(int)
        high = np.floor(query.max(axis=0) - image.min(axis=0) + padding).astype(int)
        for shift in product(*(range(lo, hi+1) for lo, hi in zip(low, high, strict=True))):
            if op.triplet() == "x,y,z" and shift == (0, 0, 0):
                continue
            delta = (image[None, :, :] + shift - query[:, None, :]) @ orth.T
            distances = np.linalg.norm(delta, axis=2)
            for query_index, atom_index in np.argwhere(distances < cutoff):
                contacts.append(dict(query_atom=int(window_indices[query_index]), mate_atom=int(atom_index),
                                     operation=op.triplet(), translation=list(shift),
                                     distance_A=float(distances[query_index, atom_index])))
    return sorted(contacts, key=lambda hit: hit["distance_A"])


def audit(source, report_path, chain_name, start, length, cutoff):
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    report = json.loads(report_path.read_bytes())
    if source_hash != report["target_sha256"]:
        raise ValueError("The coordinate source does not match the fit report.")
    row, = (w for w in report["chains"][chain_name]["windows"]
            if w["start"] == start and len(w["sequence"]) == length)
    fit = row["terminal_backbone_angle_probe"]["best"]
    if not fit["passed_context_without_second_adjustment"]:
        raise ValueError("The candidate must have passed the asymmetric-unit checks.")
    structure = gemmi.read_structure(str(source))
    group = structure.find_spacegroup()
    if group is None or not structure.cell.is_crystal():
        raise ValueError("The source requires an experimental crystal cell and space group.")
    model = structure[0]
    chain = list(model[chain_name])
    offset, = (i for i, residue in enumerate(chain) if str(residue.seqid) == start)
    window_ids = [str(r.seqid) for r in chain[offset:offset+length]]
    replacements = {(window_ids[a["residue_index"]], a["atom_name"]): a["xyz_A"]
                    for a in fit["candidate_geometry"]["atoms"]}
    deposited = []
    keys = []
    rebuilt = []
    changed = []
    for c in model:
        for residue in c:
            for atom in residue:
                if atom.element.atomic_number <= 1 or atom.occ <= 0:
                    continue
                position = [atom.pos.x, atom.pos.y, atom.pos.z]
                key = str(residue.seqid), atom.name
                if c.name == chain_name and key[0] in window_ids:
                    changed.append(len(keys))
                    rebuilt.append(replacements.pop(key))
                else:
                    rebuilt.append(position)
                keys.append((c.name, *key, str(atom.altloc)))
                deposited.append(position)
    if replacements or len(changed) != len(fit["candidate_geometry"]["atoms"]):
        raise ValueError("Candidate and deposited window atom inventories differ.")

    def measure(values):
        hits = crystal_contacts(np.asarray(values), changed, structure.cell, group.operations(), cutoff)
        return [dict(hit, query=keys[hit["query_atom"]], mate=keys[hit["mate_atom"]]) for hit in hits]

    return dict(target=report["target"], target_sha256=source_hash,
                fit_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                chain=chain_name, residues=window_ids, space_group=group.hm,
                cutoff_A=cutoff, window_heavy_atoms=len(changed),
                deposited_contacts=measure(deposited), rebuilt_contacts=measure(rebuilt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("fit_report", type=Path)
    parser.add_argument("--chain", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--length", required=True, type=int)
    parser.add_argument("--cutoff", default=2., type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.cutoff) or args.cutoff <= 0:
        parser.error("--cutoff must be finite and positive")
    result = audit(args.source, args.fit_report, args.chain, args.start, args.length, args.cutoff)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
