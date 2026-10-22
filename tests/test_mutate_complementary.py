import mdna
from mdna.utils import get_base_pair_dict


def resnames(dna):
    return [r.name for r in dna.get_traj().topology.residues]


def test_complementary_mutation_from_either_strand():
    dna = mdna.make(sequence="GACGTTCA")
    n = 8
    # residue 2 is on the leading strand; residue 2n-1-5 is the partner of bp 5, on the other strand
    dna.mutate(mutations={2: "A", 2 * n - 1 - 5: "G"}, complementary=True)
    names = resnames(dna)
    assert (names[2], names[2 * n - 1 - 2]) == ("DA", "DT")
    assert (names[2 * n - 1 - 5], names[5]) == ("DG", "DC")


def test_pairs_follow_chain_position_with_non_canonical_bases():
    dna = mdna.make(sequence="GBCZTA")
    residues = list(dna.get_traj().topology.residues)
    pairs = get_base_pair_dict(dna.get_traj())
    assert all(pairs[residues[i]] is residues[11 - i] and pairs[residues[11 - i]] is residues[i] for i in range(6))
