"""Locate remaining fixed-atom contacts and large shifts after DNA relaxation."""

import argparse
from contextlib import redirect_stdout
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

from evaluate_open_path_bends import ROOT
from probe_amber_anchor_restraints import _glycosidic_base_group
from probe_resampled_backbone_relaxation import replay


def audit(path):
    from mdna.structure_qc import measure_open_chain_geometry

    raw = path.read_bytes()
    record = json.loads(raw)
    source = path.with_name(record["source_report"])
    seed, source_hash, trajectory = replay(source)
    if seed != record["seed"] or source_hash != record["source_sha256"]:
        raise ValueError("Minimization source changed")
    trajectory = trajectory.atom_slice([
        atom.index for atom in trajectory.topology.atoms if atom.element.atomic_number > 1])
    keys = [[atom.residue.index, atom.residue.name, atom.name] for atom in trajectory.topology.atoms]
    prior = np.asarray(record["initial_heavy_xyz_nm"])
    if keys != record["atom_keys"] or not np.array_equal(trajectory.xyz[0], prior):
        raise ValueError("Heavy atom order or starting positions changed")

    topology = trajectory.topology.to_openmm()
    bonds = np.asarray([(a.index, b.index) for a, b in trajectory.topology.bonds])
    initial_bond_A = np.linalg.norm(prior[bonds[:, 0]] - prior[bonds[:, 1]], axis=1) * 10.
    fixed = set()
    for chain in topology.chains():
        residues = list(chain.residues())
        for residue in residues:
            _, base_heavy = _glycosidic_base_group(topology, residue)
            fixed.update(base_heavy)
            fixed.update(atom.index for atom in residue.atoms()
                         if atom.name == "C1'" or residue in (residues[0], residues[-1]))

    def inspect(xyz, expected_contacts):
        trajectory.xyz[0] = xyz
        qc = measure_open_chain_geometry(trajectory, include_contact_pairs=True)
        if qc["short_heavy_contacts"] != expected_contacts or not np.array_equal(xyz[sorted(fixed)], prior[sorted(fixed)]):
            raise ValueError("Saved geometry does not reproduce reported contacts and fixed atoms")
        pairs = qc["short_heavy_contact_pairs"]
        fixed_pairs = [(i, j, d) for i, j, d in pairs if i in fixed and j in fixed]
        shifts = np.linalg.norm(xyz - prior, axis=1) * 10.
        bond_A = np.linalg.norm(xyz[bonds[:, 0]] - xyz[bonds[:, 1]], axis=1) * 10.
        bond_changes = abs(bond_A - initial_bond_A)
        return dict(total_contacts=len(pairs), fixed_fixed_contacts=len(fixed_pairs),
                    closest_fixed_fixed=[dict(atom_i=keys[i], atom_j=keys[j], distance_A=d)
                                         for i, j, d in sorted(fixed_pairs, key=lambda pair: pair[2])[:5]],
                    heavy_shift_over_1A=int(np.count_nonzero(shifts > 1.)),
                    heavy_shift_over_2A=int(np.count_nonzero(shifts > 2.)),
                    largest_heavy_shifts=[dict(atom=keys[i], shift_A=float(shifts[i]))
                                          for i in np.argsort(-shifts)[:5] if shifts[i] > 0.],
                    largest_heavy_bond_changes=[
                        dict(atom_i=keys[bonds[i, 0]], atom_j=keys[bonds[i, 1]],
                             before_A=float(initial_bond_A[i]), after_A=float(bond_A[i]))
                        for i in np.argsort(-bond_changes)[:5] if bond_changes[i] > 0.])

    return dict(seed=seed, input_report=path.name, input_sha256=sha256(raw).hexdigest(),
                initial=inspect(prior, record["initial"]["short_heavy_contacts"]), results=[
                    dict(restraint_k_kJ_mol_nm2=row["restraint_k_kJ_mol_nm2"],
                         **inspect(np.asarray(row["heavy_xyz_nm"]), row["geometry"]["short_heavy_contacts"]))
                    for row in record["results"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    with redirect_stdout(sys.stderr):
        result = [audit(path) for path in args.source]
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
