"""Measure open-chain covalent geometry and short heavy-atom contacts."""

import numpy as np
from scipy.spatial import cKDTree


def _angle(a, b, c):
    left, right = a - b, c - b
    if min(np.linalg.norm(left), np.linalg.norm(right)) <= 1e-12:
        return None
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(left, right)), left @ right)))


def _dihedral(a, b, c, d):
    axis = c - b
    length = np.linalg.norm(axis)
    if length <= 1e-12:
        return None
    axis /= length
    first, last = a - b, d - c
    first -= (first @ axis) * axis
    last -= (last @ axis) * axis
    if min(np.linalg.norm(first), np.linalg.norm(last)) <= 1e-12:
        return None
    return float(np.degrees(np.arctan2(np.cross(axis, first) @ last, first @ last)))


def measure_link_coordinates(left, right):
    """Measure a directed link from two atom-name→xyz maps in angstroms."""
    metrics = {}
    definitions = {
        'o3p_A': (_distance, ((left, "O3'"), (right, "P"))),
        'c3_o3_p_deg': (_angle, ((left, "C3'"), (left, "O3'"), (right, "P"))),
        'o3_p_o5_deg': (_angle, ((left, "O3'"), (right, "P"), (right, "O5'"))),
        'p_o5_c5_deg': (_angle, ((right, "P"), (right, "O5'"), (right, "C5'"))),
        'c4_c3_o3_p_deg': (_dihedral, ((left, "C4'"), (left, "C3'"), (left, "O3'"), (right, "P"))),
        'c3_o3_p_o5_deg': (_dihedral, ((left, "C3'"), (left, "O3'"), (right, "P"), (right, "O5'"))),
        'o3_p_o5_c5_deg': (_dihedral, ((left, "O3'"), (right, "P"), (right, "O5'"), (right, "C5'"))),
        'p_o5_c5_c4_deg': (_dihedral, ((right, "P"), (right, "O5'"), (right, "C5'"), (right, "C4'"))),
    }
    for name, (measure, atoms) in definitions.items():
        metrics[name] = (measure(*(np.asarray(coords[atom], dtype=float) for coords, atom in atoms))
                         if all(atom in coords for coords, atom in atoms) else None)
    return metrics


def _distance(a, b):
    return float(np.linalg.norm(a - b))


def measure_open_chain_geometry(trajectory, *, frame=0, contact_cutoff_A=2.0, include_contact_pairs=False):
    """Measure a nucleic-acid MDTraj trajectory; input coordinates are nm.

    All topology chains must be open nucleic-acid polymers in 5' to 3' order.
    Contacts exclude atom pairs separated by up to three topology bonds. This
    is a fixed-distance screen and does not assign a MolProbity clashscore.
    When requested, return contact pairs as two topology atom indices and a distance in Å.
    """
    xyz = np.asarray(trajectory.xyz[frame], dtype=float) * 10
    atoms = list(trajectory.topology.atoms)
    cutoff = float(contact_cutoff_A)
    if xyz.shape != (len(atoms), 3) or not len(atoms) or not np.isfinite(xyz).all():
        raise ValueError('expected finite xyz with one coordinate per topology atom')
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError('contact_cutoff_A must be finite and positive')
    neighbors = [set() for _ in atoms]
    for a, b in trajectory.topology.bonds:
        neighbors[a.index].add(b.index)
        neighbors[b.index].add(a.index)
    links = []
    for chain in trajectory.topology.chains:
        residues = list(chain.residues)
        maps = [{a.name: a.index for a in res.atoms} for res in residues]
        if any(not {"C3'", "C4'"} <= names.keys() for names in maps):
            raise ValueError('expected nucleic-acid residues with standard sugar atom names')
        if maps and "O3'" in maps[-1] and 'P' in maps[0]:
            if maps[0]['P'] in neighbors[maps[-1]["O3'"]]:
                raise ValueError('expected an open chain; found a terminal O3\'-P bond')
        for left, right, lmap, rmap in zip(residues, residues[1:], maps, maps[1:]):
            row = dict(chain_index=chain.index, left_residue_index=left.index,
                       right_residue_index=right.index, bond_present=False,
                       left_residue_name=getattr(left, 'name', None),
                       right_residue_name=getattr(right, 'name', None))
            row.update(measure_link_coordinates(
                {name: xyz[i] for name, i in lmap.items()},
                {name: xyz[i] for name, i in rmap.items()}))
            if "O3'" in lmap and 'P' in rmap:
                o3, p = lmap["O3'"], rmap['P']
                row['bond_present'] = p in neighbors[o3]
            links.append(row)
    heavy = [a.index for a in atoms if a.element is not None and a.element.atomic_number > 1]
    pairs = cKDTree(xyz[heavy]).query_pairs(cutoff, output_type='ndarray')
    near = []
    for i0, j0 in pairs:
        i, j = heavy[i0], heavy[j0]
        if j in neighbors[i] or any(j in neighbors[k] or neighbors[j] & neighbors[k]
                                    for k in neighbors[i]):
            continue
        distance = float(np.linalg.norm(xyz[i] - xyz[j]))
        if distance < cutoff:
            near.append((i, j, distance))
    result = dict(links=links, missing_links=sum(not row['bond_present'] for row in links),
                  contact_cutoff_A=cutoff, short_heavy_contacts=len(near),
                  min_short_heavy_distance_A=min((row[2] for row in near), default=None),
                  atoms_without_element=sum(a.element is None for a in atoms))
    if include_contact_pairs:
        result['short_heavy_contact_pairs'] = sorted(near)
    return result
