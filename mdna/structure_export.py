"""Coordinate and covalent graph export for assembled standard RNA/DNA."""
from collections import defaultdict
from pathlib import Path

from Bio.PDB import MMCIFIO
import numpy as np


def save_fragment_mmcif(assembly, filename, *, polymer_type):
    """Write one assembled polymer; use the validated assembly as the bond source."""
    trajectory = assembly.to_mdtraj()
    return _save_mmcif(trajectory, assembly.atoms, assembly.bonds, filename, polymer_type)


def save_standard_dna_mmcif(trajectory, filename, *, frame=-1):
    """Write a standard DNA model with explicit topology edges."""
    residues = list(trajectory.topology.residues)
    if not residues or any(r.name not in {"DA", "DC", "DG", "DT"} for r in residues):
        raise ValueError("DNA mmCIF export requires standard DA/DC/DG/DT residues.")
    xyz = np.asarray(trajectory.xyz[frame], dtype=float) * 10.
    if not np.isfinite(xyz).all():
        raise ValueError("DNA mmCIF export requires finite coordinates.")
    atoms = []
    for atom in trajectory.topology.atoms:
        if atom.element is None or atom.element.atomic_number <= 0:
            raise ValueError("DNA mmCIF export requires a chemical element for each atom.")
        atoms.append(dict(residue_index=atom.residue.index, atom_name=atom.name,
                          element=atom.element.symbol, xyz_A=xyz[atom.index]))
    keys = {(a["residue_index"], a["atom_name"]) for a in atoms}
    if len(keys) != len(atoms):
        raise ValueError("DNA mmCIF export requires unique atom names within each residue.")
    orders = {1: "single", 2: "double", 3: "triple", 4: "quadruple"}
    bonds = [dict(residue_index_1=a.residue.index, atom_1=a.name,
                  residue_index_2=b.residue.index, atom_2=b.name,
                  order=orders.get(getattr(bond, "order", None), "unspecified"))
             for bond in trajectory.topology.bonds for a, b in [bond]]
    return _save_mmcif(trajectory, atoms, bonds, filename, "polydeoxyribonucleotide")


def _chain_label(index):
    result = ""
    while index >= 0:
        index, digit = divmod(index, 26)
        result = chr(65+digit) + result
        index -= 1
    return result


def _save_mmcif(trajectory, atoms, bonds, filename, polymer_type):
    residues = list(trajectory.topology.residues)
    atoms = sorted(atoms, key=lambda a: (a["residue_index"], a["atom_name"]))
    data = defaultdict(list)
    data["data_"] = "mdna_fragment"
    data["_chem_comp.id"] = sorted({r.name for r in residues})
    entities = {}
    locations = {}
    for chain in trajectory.topology.chains:
        label = _chain_label(chain.index)
        chain_residues = list(chain.residues)
        sequence = tuple(r.name for r in chain_residues)
        if sequence not in entities:
            entity = str(len(entities)+1)
            entities[sequence] = entity
            data["_entity.id"].append(entity)
            data["_entity.type"].append("polymer")
            data["_entity_poly.entity_id"].append(entity)
            data["_entity_poly.type"].append(polymer_type)
            for i, name in enumerate(sequence, 1):
                data["_entity_poly_seq.entity_id"].append(entity)
                data["_entity_poly_seq.num"].append(str(i))
                data["_entity_poly_seq.mon_id"].append(name)
                data["_entity_poly_seq.hetero"].append("n")
        entity = entities[sequence]
        data["_struct_asym.id"].append(label)
        data["_struct_asym.entity_id"].append(entity)
        for i, r in enumerate(chain_residues, 1):
            locations[r.index] = (label, i, entity, r.name)
    for i, atom in enumerate(atoms, 1):
        label, seq, entity, res = locations[atom["residue_index"]]
        name = atom["atom_name"]
        x, y, z = atom["xyz_A"]
        row = dict(group_PDB="ATOM", id=i, type_symbol=atom["element"], label_atom_id=name,
                   label_alt_id=".", label_comp_id=res, label_asym_id=label, label_entity_id=entity,
                   label_seq_id=seq, pdbx_PDB_ins_code="?", Cartn_x=f"{x:.6f}",
                   Cartn_y=f"{y:.6f}", Cartn_z=f"{z:.6f}", occupancy="1.00",
                   B_iso_or_equiv="0.00", auth_seq_id=seq, auth_asym_id=label,
                   auth_comp_id=res, auth_atom_id=name, pdbx_PDB_model_num=1)
        for key, value in row.items():
            data["_atom_site."+key].append(str(value))
    # The PDBx struct_conn order vocabulary has no aromatic value.
    order = {"single": "sing", "double": "doub", "triple": "trip", "quadruple": "quad"}
    if bonds:
        data["_struct_conn_type.id"] = "covale"
    for i, bond in enumerate(bonds, 1):
        row = dict(id=f"bond{i}", conn_type_id="covale",
                   pdbx_value_order=order.get(bond["order"], "?"))
        for n in (1, 2):
            label, seq, _, name = locations[bond[f"residue_index_{n}"]]
            row.update({f"ptnr{n}_label_asym_id": label, f"ptnr{n}_label_seq_id": seq,
                        f"ptnr{n}_label_comp_id": name, f"ptnr{n}_label_atom_id": bond[f"atom_{n}"],
                        f"ptnr{n}_auth_asym_id": label, f"ptnr{n}_auth_seq_id": seq,
                        f"ptnr{n}_auth_comp_id": name, f"ptnr{n}_symmetry": "1_555"})
        for key, value in row.items():
            data["_struct_conn."+key].append(str(value))
    path = Path(filename)
    writer = MMCIFIO()
    writer.set_dict(dict(data))
    writer.save(str(path))
    return path
