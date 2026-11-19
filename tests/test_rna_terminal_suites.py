import numpy as np
import pytest

from mdna.chemistry import derive_rna_component_state
from scripts.rna_pilot.extract_rna_terminal_suites import extract_terminal_suite
from scripts.rna_pilot.extract_rna_suite_library import validate_suite


class _FakeElement:
    def __init__(self, name):
        self.name = name


class _FakePosition:
    def __init__(self, xyz):
        self.x, self.y, self.z = map(float, xyz)


class _FakeAtom:
    def __init__(self, definition, translation):
        self.name = definition.name
        self.element = _FakeElement(definition.element)
        xyz = np.asarray(definition.ideal_coordinate, dtype=float) + translation
        self.pos = _FakePosition(xyz)
        self.occ = 1.0
        self.altloc = " "


class _FakeResidue:
    def __init__(self, base, sequence_id, state, translation):
        self.name = base
        self.seqid = sequence_id
        self._atoms = [
            _FakeAtom(atom, translation)
            for atom in state.atoms
            if atom.element.upper() != "H" and atom.ideal_coordinate is not None
        ]

    def __iter__(self):
        return iter(self._atoms)


def _terminal_suite(role):
    if role == "five_prime":
        first_state = derive_rna_component_state(
            "RNA:U", five_prime="hydroxyl", three_prime="phosphodiester"
        )
        second_state = derive_rna_component_state(
            "RNA:C", five_prime="phosphodiester", three_prime="phosphodiester"
        )
        first, second = (
            _FakeResidue("U", 1, first_state, np.array([0.0, 0.0, 0.0])),
            _FakeResidue("C", 2, second_state, np.array([8.0, 0.0, 0.0])),
        )
    else:
        first_state = derive_rna_component_state(
            "RNA:G", five_prime="phosphodiester", three_prime="phosphodiester"
        )
        second_state = derive_rna_component_state(
            "RNA:A", five_prime="phosphodiester", three_prime="hydroxyl"
        )
        first, second = (
            _FakeResidue("G", 7, first_state, np.array([0.0, 0.0, 0.0])),
            _FakeResidue("A", 8, second_state, np.array([8.0, 0.0, 0.0])),
        )
    return first, second


@pytest.mark.parametrize(
    "role,expected_sequence,expected_states",
    [
        ("five_prime", "UC", (
            "5prime_hydroxyl+3prime_phosphodiester",
            "5prime_phosphodiester+3prime_phosphodiester",
        )),
        ("three_prime", "GA", (
            "5prime_phosphodiester+3prime_phosphodiester",
            "5prime_phosphodiester+3prime_hydroxyl",
        )),
    ],
)
def test_terminal_suite_uses_observed_end_chemistry_and_matching_heavy_graph(
    role, expected_sequence, expected_states
):
    upstream, downstream = _terminal_suite(role)

    suite = extract_terminal_suite(
        "TEST", "A", upstream, downstream, "a" * 64, "UCGUACGA", role
    )

    assert suite["terminal_role"] == role
    assert suite["sequence"] == expected_sequence
    assert tuple(suite["residue_states"]) == expected_states
    validate_suite(suite)
    names_by_residue = {
        index: {atom["atom_name"] for atom in suite["atoms"] if atom["residue_index"] == index}
        for index in (0, 1)
    }
    if role == "five_prime":
        assert not ({"P", "OP1", "OP2"} & names_by_residue[0])
        assert "P" in names_by_residue[1]
    else:
        assert "P" in names_by_residue[0]
        assert "P" in names_by_residue[1]


def test_terminal_suite_rejects_unknown_role():
    upstream, downstream = _terminal_suite("five_prime")

    with pytest.raises(ValueError, match="terminal_role"):
        extract_terminal_suite(
            "TEST", "A", upstream, downstream, "a" * 64, "UCGUACGA", "internal"
        )
