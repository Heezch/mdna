import mdna as mdna
import pytest

def test_make():
    n_bp = 10
    dna = mdna.make(n_bp=n_bp)
    assert len(dna.sequence) == n_bp


def test_sequence_to_pdb_keeps_terminal_phosphate_for_circular():
    sequence = "A" * 20
    circular_shape = mdna.Shapes.circle(radius=1)

    linear = mdna.sequence_to_pdb(
        sequence=sequence,
        save=False,
        output='GROMACS',
        shape=circular_shape,
        circular=False,
    )
    circular = mdna.sequence_to_pdb(
        sequence=sequence,
        save=False,
        output='GROMACS',
        shape=circular_shape,
        circular=True,
    )

    linear_terminal = list(linear.top.chain(0).residues)[0]
    circular_terminal = list(circular.top.chain(0).residues)[0]

    linear_atoms = {atom.name for atom in linear_terminal.atoms}
    circular_atoms = {atom.name for atom in circular_terminal.atoms}

    assert {'P', 'OP1', 'OP2'}.isdisjoint(linear_atoms)
    assert {'P', 'OP1', 'OP2'}.issubset(circular_atoms)
    assert circular.n_atoms == linear.n_atoms + 6


def test_make_circular_get_traj_preserves_terminal_phosphate():
    dna = mdna.make(sequence="A" * 20, circular=True)
    traj = dna.get_traj()
    terminal_residue = list(traj.top.chain(0).residues)[0]
    atom_names = {atom.name for atom in terminal_residue.atoms}

    assert {'P', 'OP1', 'OP2'}.issubset(atom_names)


def test_make_without_control_points_uses_sequence_length_linear_and_circular():
    sequence = "ATGCATGCATGC"

    linear = mdna.make(sequence=sequence, circular=False)
    circular = mdna.make(sequence=sequence, circular=True)

    assert linear.n_bp == len(sequence)
    assert circular.n_bp == len(sequence)
    assert linear.sequence == sequence
    assert circular.sequence == sequence


def test_make_without_control_points_and_only_nbp_generates_random_sequence():
    n_bp = 18
    dna = mdna.make(n_bp=n_bp)

    assert dna.n_bp == n_bp
    assert len(dna.sequence) == n_bp


def test_make_with_control_points_and_no_sequence_no_nbp_infers_length_for_four_points():
    control_points = mdna.Shapes.line(length=1, num_points=4)
    dna = mdna.make(control_points=control_points)

    assert dna.n_bp == dna.frames.shape[0]
    assert len(dna.sequence) == dna.n_bp


def test_make_with_control_points_and_no_sequence_no_nbp_infers_length_for_many_points():
    control_points = mdna.Shapes.line(length=1, num_points=8)
    dna = mdna.make(control_points=control_points)

    assert dna.n_bp == dna.frames.shape[0]
    assert len(dna.sequence) == dna.n_bp


def test_make_with_control_points_and_sequence_scales_to_sequence_length():
    control_points = mdna.Shapes.circle(radius=1, num_points=10)
    sequence = "A" * 30
    dna = mdna.make(sequence=sequence, control_points=control_points, circular=True)

    assert dna.n_bp == len(sequence)
    assert dna.sequence == sequence


def test_make_with_control_points_and_nbp_scales_and_generates_random_sequence():
    control_points = mdna.Shapes.circle(radius=1, num_points=10)
    n_bp = 24
    dna = mdna.make(control_points=control_points, n_bp=n_bp, circular=True)

    assert dna.n_bp == n_bp
    assert len(dna.sequence) == n_bp


def test_make_closed_argument_is_deprecated_alias_for_circular():
    with pytest.warns(DeprecationWarning):
        dna = mdna.make(sequence="A" * 20, closed=True)

    assert dna.circular is True


def test_make_rejects_too_few_control_points():
    with pytest.raises(ValueError):
        mdna.make(control_points=mdna.Shapes.line(length=1, num_points=3))


def test_make_rejects_sequence_nbp_mismatch():
    with pytest.raises(ValueError):
        mdna.make(sequence="AAAA", n_bp=5)


def test_make_rejects_non_positive_nbp():
    with pytest.raises(ValueError):
        mdna.make(n_bp=0)
    with pytest.raises(ValueError):
        mdna.make(n_bp=-1)


def test_make_accepts_sequence_as_list():
    dna = mdna.make(sequence=["A", "T", "G", "C"])
    assert dna.sequence == "ATGC"
    assert dna.n_bp == 4


def test_make_rejects_control_points_with_invalid_shape():
    with pytest.raises(ValueError):
        mdna.make(control_points=[[0, 0], [1, 0], [2, 0], [3, 0]])
    with pytest.raises(ValueError):
        mdna.make(control_points=[0, 1, 2, 3])


def test_sequence_to_pdb_non_gromacs_keeps_terminal_phosphates_linear():
    sequence = "A" * 20
    traj = mdna.sequence_to_pdb(sequence=sequence, save=False, output='PDB', circular=False)

    terminal = list(traj.top.chain(0).residues)[0]
    atom_names = {atom.name for atom in terminal.atoms}
    assert {'P', 'OP1', 'OP2'}.issubset(atom_names)