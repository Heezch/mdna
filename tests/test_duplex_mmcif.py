import numpy as np
from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

import mdna


def test_standard_duplex_mmcif_preserves_chain_order_xyz_and_bonds(tmp_path):
    dna = mdna.make(sequence="AACG")
    trajectory = dna.get_traj()
    path = dna.save_mmcif(tmp_path / "duplex.cif")
    raw = MMCIF2Dict(str(path))
    assert raw["_struct_asym.id"] == ["A", "B"]
    assert len(set(raw["_struct_asym.entity_id"])) == 2
    structure = MMCIFParser(QUIET=True).get_structure("duplex", path)
    chains = {c.id: c for c in structure.get_chains()}
    assert [[r.resname for r in chains[k]] for k in "AB"] == [
        ["DA", "DA", "DC", "DG"], ["DC", "DG", "DT", "DT"]]
    locations = {}
    for i, chain in enumerate(trajectory.topology.chains):
        label = "AB"[i]
        for j, res in enumerate(chain.residues, 1):
            for atom in res.atoms:
                locations[atom.index] = (label, str(j), atom.name)
                np.testing.assert_allclose(chains[label][j][atom.name].coord,
                                           trajectory.xyz[0, atom.index]*10, atol=3e-6, rtol=0)
    fields = [f"_struct_conn.ptnr{n}_label_{field}" for n in (1, 2)
              for field in ("asym_id", "seq_id", "atom_id")]
    listed = list(zip(*(raw[f] for f in fields), strict=True))
    assert len(listed) == trajectory.topology.n_bonds
    assert {frozenset((r[:3], r[3:])) for r in listed} == {
        frozenset((locations[a.index], locations[b.index])) for a, b in trajectory.topology.bonds}


def test_identical_strand_sequences_share_one_entity(tmp_path):
    path = mdna.make(sequence="ACGT").save_mmcif(tmp_path / "palindrome.cif")
    raw = MMCIF2Dict(str(path))
    assert raw["_struct_asym.id"] == ["A", "B"]
    assert raw["_struct_asym.entity_id"] == ["1", "1"]
    assert raw["_entity_poly_seq.mon_id"] == ["DA", "DC", "DG", "DT"]


def check_gemmi_duplex(tmp_path):
    import gemmi
    dna = mdna.make(sequence="AACG")
    traj = dna.get_traj()
    path = dna.save_mmcif(tmp_path / "duplex.cif")
    model = gemmi.read_structure(str(path))
    assert [chain.name for chain in model[0]] == ["A", "B"]
    assert sum(len(res) for chain in model[0] for res in chain) == traj.n_atoms
    assert len(model.connections) == traj.topology.n_bonds
    links = [c for c in model.connections if c.partner1.res_id.seqid != c.partner2.res_id.seqid]
    assert len(links) == 6
    assert {c.partner1.chain_name for c in links} == {"A", "B"}


if __name__ == "__main__":
    from pathlib import Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as run_dir:
        check_gemmi_duplex(Path(run_dir))
    print("Gemmi duplex round-trip passed")
