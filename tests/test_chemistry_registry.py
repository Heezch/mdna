"""Behavioral contract for explicit nucleic-acid components and links."""

import pytest

from mdna.chemistry import (
    AtomDefinition,
    BondDefinition,
    ComponentDefinition,
    ComponentRegistry,
    LinkDefinition,
    NUCLEOTIDE_REGISTRY,
    get_component_definition,
    get_nucleotide_definition,
)


def _component(component_id, polymer_type, sugar, residue_name, atom_names):
    atoms = tuple(
        AtomDefinition(name=name, element="P" if name == "P" else "C")
        for name in atom_names
    )
    bonds = tuple(
        BondDefinition(atom_1=left, atom_2=right, order="single")
        for left, right in zip(atom_names, atom_names[1:])
    )
    return ComponentDefinition(
        component_id=component_id,
        polymer_type=polymer_type,
        sugar_chemistry=sugar,
        backbone_chemistry="phosphodiester",
        residue_name=residue_name,
        atoms=atoms,
        bonds=bonds,
        source="https://example.org/component-definition",
        sha256="a" * 64,
    )


def test_existing_nucleotide_registry_remains_compatible():
    assert len(NUCLEOTIDE_REGISTRY) == 13
    definition = get_nucleotide_definition("b")
    assert definition.component_id == "IGU"
    assert definition.template_key == "P"
    assert definition.complement == "S"


def test_all_legacy_codes_resolve_to_source_hashed_component_graphs():
    for legacy in NUCLEOTIDE_REGISTRY.values():
        component = get_component_definition(f"DNA:{legacy.component_id}")
        assert component.polymer_type == "DNA"
        assert component.residue_name == legacy.residue_name
        assert component.sugar_chemistry == legacy.sugar_chemistry
        assert component.atoms
        assert component.bonds
        assert len(component.sha256) == 64


def test_deoxyuridine_template_is_not_registered_as_rna_uridine():
    deoxyuridine = get_component_definition("DNA:DU")
    ribouridine = get_component_definition("RNA:U")
    assert deoxyuridine.sugar_chemistry == "2-deoxyribose"
    assert ribouridine.sugar_chemistry == "ribose"
    assert deoxyuridine.component_id != ribouridine.component_id
    assert "O2'" not in {atom.name for atom in deoxyuridine.atoms}
    assert "O2'" in {atom.name for atom in ribouridine.atoms}


def test_official_rna_ccd_components_are_registered_with_ribose_and_provenance():
    expected = {
        "A": (37, 39, "46db841d33a8147dcd5a3802123e362d708c875f9ba7db457525f094ab55179a"),
        "C": (35, 36, "4937647bc2142b3a67fedb3db7e4f24c0ecad844ca5330356f2e95a981e9544e"),
        "G": (38, 40, "4fbcf5b5222d7c7f0a761676cffa9bb997daa8f76d41fdb7657aa59e969b85fa"),
        "U": (34, 35, "de78a1e4a9860baaefc17c5e509fad96b0b25e0b62f90352976f9079cd4b051b"),
    }
    for code, (atom_count, bond_count, source_hash) in expected.items():
        component = get_component_definition(f"RNA:{code}")
        assert component.polymer_type == "RNA"
        assert component.sugar_chemistry == "ribose"
        assert component.residue_name == code
        assert len(component.atoms) == atom_count
        assert len(component.bonds) == bond_count
        assert "O2'" in {atom.name for atom in component.atoms}
        assert all(atom.ideal_coordinate is not None for atom in component.atoms)
        assert component.coordinate_unit == "angstrom"
        assert component.sha256 == source_hash
        assert "files.rcsb.org/ligands/download/" in component.source


def test_component_registry_keeps_dna_and_rna_chemistry_distinct():
    registry = ComponentRegistry()
    dna_a = _component(
        "DNA:DA", "DNA", "2-deoxyribose", "DA", ("C1'", "O3'", "P")
    )
    rna_a = _component(
        "RNA:A", "RNA", "ribose", "A", ("C1'", "O2'", "O3'", "P")
    )

    registry.register(dna_a)
    registry.register(rna_a)

    assert registry.get("DNA:DA").sugar_chemistry == "2-deoxyribose"
    assert registry.get("RNA:A").sugar_chemistry == "ribose"
    assert "O2'" not in {atom.name for atom in registry.get("DNA:DA").atoms}
    assert "O2'" in {atom.name for atom in registry.get("RNA:A").atoms}


def test_unknown_component_is_rejected():
    with pytest.raises(ValueError, match="Unknown component"):
        ComponentRegistry().get("RNA:PSU")


def test_component_rejects_a_bond_with_an_unknown_atom():
    with pytest.raises(ValueError, match="bond endpoint"):
        ComponentDefinition(
            component_id="RNA:A",
            polymer_type="RNA",
            sugar_chemistry="ribose",
            backbone_chemistry="phosphodiester",
            residue_name="A",
            atoms=(AtomDefinition(name="O2'", element="O"),),
            bonds=(BondDefinition("O2'", "P", "single"),),
            source="https://example.org/component-definition",
            sha256="a" * 64,
        )


def test_atom_rejects_non_finite_ideal_coordinates():
    with pytest.raises(ValueError, match="finite"):
        AtomDefinition(name="O2'", element="O", ideal_coordinate=(0.0, float("nan"), 1.0))


def test_link_definition_validates_ports_and_leaving_atoms():
    registry = ComponentRegistry()
    upstream = _component(
        "RNA:A", "RNA", "ribose", "A", ("C1'", "O3'", "HO3'")
    )
    downstream = _component(
        "RNA:C", "RNA", "ribose", "C", ("C1'", "P", "OP3")
    )
    registry.register(upstream)
    registry.register(downstream)
    link = LinkDefinition(
        link_id="RNA:3to5",
        upstream_component_id="RNA:A",
        downstream_component_id="RNA:C",
        upstream_port="O3'",
        downstream_port="P",
        bond_order="single",
        remove_upstream_atoms=("HO3'",),
        remove_downstream_atoms=("OP3",),
        source="https://example.org/polymer-link",
        evidence_sha256="b" * 64,
    )

    registry.validate_link(link)


def test_link_definition_rejects_a_port_missing_from_its_component():
    registry = ComponentRegistry()
    upstream = _component("RNA:A", "RNA", "ribose", "A", ("C1'", "O3'"))
    downstream = _component("RNA:C", "RNA", "ribose", "C", ("C1'", "P"))
    registry.register(upstream)
    registry.register(downstream)
    link = LinkDefinition(
        link_id="RNA:3to5",
        upstream_component_id="RNA:A",
        downstream_component_id="RNA:C",
        upstream_port="O3'",
        downstream_port="OP3",
        bond_order="single",
        source="https://example.org/polymer-link",
        evidence_sha256="b" * 64,
    )

    with pytest.raises(ValueError, match="downstream port"):
        registry.validate_link(link)


def test_standard_rna_3to5_links_are_registered_with_observed_leaving_atoms():
    from mdna.chemistry import get_link_definition

    link = get_link_definition("RNA:A->RNA:C:3to5")
    assert link.upstream_port == "O3'"
    assert link.downstream_port == "P"
    assert link.remove_upstream_atoms == ("HO3'",)
    assert link.remove_downstream_atoms == ("OP3", "HOP3")
    assert "4U3R.audit.json" in link.source
    assert link.evidence_sha256 == "2e4eb14d82021b925e1aafef4263842f09ba35489c58d07f5978ff0a8c4eb105"


def test_compile_polymer_graph_applies_link_leaving_atoms_and_cross_residue_bonds():
    from mdna.chemistry import compile_polymer_graph

    graph = compile_polymer_graph(
        ("RNA:A", "RNA:C"),
        ("RNA:A->RNA:C:3to5",),
    )

    assert graph.component_ids == ("RNA:A", "RNA:C")
    atom_keys = {atom.key for atom in graph.atoms}
    assert (0, "O3'") in atom_keys
    assert (0, "HO3'") not in atom_keys
    assert (1, "P") in atom_keys
    assert (1, "OP3") not in atom_keys
    assert (1, "HOP3") not in atom_keys
    assert (0, "O3'", 1, "P", "single") in graph.bonds


def test_compile_polymer_graph_rejects_link_with_wrong_component_order():
    from mdna.chemistry import compile_polymer_graph

    with pytest.raises(ValueError, match="does not connect"):
        compile_polymer_graph(
            ("RNA:C", "RNA:A"),
            ("RNA:A->RNA:C:3to5",),
        )


def test_compile_polymer_graph_requires_one_link_per_adjacent_residue_pair():
    from mdna.chemistry import compile_polymer_graph

    with pytest.raises(ValueError, match="one link per adjacent"):
        compile_polymer_graph(("RNA:A", "RNA:C"), ())


def test_all_standard_rna_pairwise_links_compile_the_declared_ports():
    from itertools import product

    from mdna.chemistry import compile_polymer_graph, get_link_definition

    for upstream, downstream in product("ACGU", repeat=2):
        link_id = f"RNA:{upstream}->RNA:{downstream}:3to5"
        link = get_link_definition(link_id)
        graph = compile_polymer_graph(
            (f"RNA:{upstream}", f"RNA:{downstream}"), (link_id,)
        )
        assert (0, link.upstream_port, 1, link.downstream_port, link.bond_order) in graph.bonds
        atom_keys = {atom.key for atom in graph.atoms}
        assert all((0, atom_name) not in atom_keys for atom_name in link.remove_upstream_atoms)
        assert all((1, atom_name) not in atom_keys for atom_name in link.remove_downstream_atoms)


def test_compiled_polymer_atoms_preserve_component_element_and_coordinates():
    from mdna.chemistry import compile_polymer_graph, get_component_definition

    graph = compile_polymer_graph(
        ("RNA:A", "RNA:C"),
        ("RNA:A->RNA:C:3to5",),
    )
    upstream_o3 = next(atom for atom in graph.atoms if atom.key == (0, "O3'"))
    source_o3 = next(
        atom
        for atom in get_component_definition("RNA:A").atoms
        if atom.name == "O3'"
    )

    assert upstream_o3.definition == source_o3
    assert upstream_o3.definition.element == "O"
    assert upstream_o3.definition.ideal_coordinate is not None


def test_rna_5prime_hydroxyl_state_replaces_ccd_phosphate_with_unplaced_ho5():
    from mdna.chemistry import derive_rna_component_state

    state = derive_rna_component_state(
        "RNA:A", five_prime="hydroxyl", three_prime="hydroxyl"
    )
    atom_map = {atom.name: atom for atom in state.atoms}
    bond_pairs = {frozenset((bond.atom_1, bond.atom_2)) for bond in state.bonds}

    assert state.state_id == "5prime_hydroxyl+3prime_hydroxyl"
    assert not {"P", "OP1", "OP2", "OP3", "HOP2", "HOP3"} & atom_map.keys()
    assert "HO5'" in atom_map
    assert "HO3'" in atom_map
    assert frozenset(("O5'", "HO5'")) in bond_pairs
    assert atom_map["HO5'"].ideal_coordinate is None
    assert state.coordinate_status == "incomplete"
    assert "4U3R.audit.json" in state.source


def test_rna_internal_state_removes_linking_and_phosphate_protons():
    from mdna.chemistry import derive_rna_component_state

    state = derive_rna_component_state(
        "RNA:C", five_prime="phosphodiester", three_prime="phosphodiester"
    )
    atom_names = {atom.name for atom in state.atoms}

    assert state.state_id == "5prime_phosphodiester+3prime_phosphodiester"
    assert {"P", "OP1", "OP2", "O2'", "O3'"} <= atom_names
    assert not {"OP3", "HOP2", "HOP3", "HO3'", "HO5'"} & atom_names
    assert state.coordinate_status == "complete_from_component"


def test_rna_three_prime_hydroxyl_keeps_observed_oh_and_removes_op3_group():
    from mdna.chemistry import derive_rna_component_state

    state = derive_rna_component_state(
        "RNA:G", five_prime="phosphodiester", three_prime="hydroxyl"
    )
    atom_names = {atom.name for atom in state.atoms}
    bond_pairs = {frozenset((bond.atom_1, bond.atom_2)) for bond in state.bonds}

    assert {"P", "OP1", "OP2", "HO3'"} <= atom_names
    assert not {"OP3", "HOP2", "HOP3", "HO5'"} & atom_names
    assert frozenset(("C3'", "O3'")) in bond_pairs
    assert state.coordinate_status == "complete_from_component"


def test_rna_component_state_rejects_unknown_end_chemistry():
    from mdna.chemistry import derive_rna_component_state

    with pytest.raises(ValueError, match="Unsupported RNA 5-prime state"):
        derive_rna_component_state(
            "RNA:U", five_prime="capped", three_prime="hydroxyl"
        )


def test_all_standard_rna_bases_support_observed_terminal_and_linkage_states():
    from mdna.chemistry import derive_rna_component_state

    state_options = (
        ("hydroxyl", "phosphodiester", {"HO5'"}, {"P", "HO3'", "OP3"}),
        ("phosphodiester", "phosphodiester", {"P", "OP1", "OP2"}, {"OP3", "HOP2", "HOP3", "HO3'", "HO5'"}),
        ("phosphodiester", "hydroxyl", {"P", "OP1", "OP2", "HO3'"}, {"OP3", "HOP2", "HOP3", "HO5'"}),
        ("monophosphate", "hydroxyl", {"P", "OP1", "OP2", "OP3", "HOP2", "HOP3", "HO3'"}, {"HO5'"}),
    )
    for base in "ACGU":
        for five_prime, three_prime, required, absent in state_options:
            state = derive_rna_component_state(
                f"RNA:{base}", five_prime=five_prime, three_prime=three_prime
            )
            atom_names = {atom.name for atom in state.atoms}
            assert required <= atom_names
            assert not absent & atom_names
            assert state.component_sha256
            assert state.evidence_sha256 == "2e4eb14d82021b925e1aafef4263842f09ba35489c58d07f5978ff0a8c4eb105"


def test_compile_rna_polymer_sequence_applies_first_internal_and_last_states():
    from mdna.chemistry import compile_rna_polymer_sequence

    graph = compile_rna_polymer_sequence("UCGUACGA")
    atom_keys = {atom.key for atom in graph.atoms}
    expected_links = {
        (index, "O3'", index + 1, "P", "single") for index in range(7)
    }

    assert graph.component_ids == tuple(f"RNA:{base}" for base in "UCGUACGA")
    assert graph.residue_state_ids[0] == "5prime_hydroxyl+3prime_phosphodiester"
    assert graph.residue_state_ids[1:-1] == (
        "5prime_phosphodiester+3prime_phosphodiester",
    ) * 6
    assert graph.residue_state_ids[-1] == "5prime_phosphodiester+3prime_hydroxyl"
    assert expected_links <= set(graph.bonds)
    assert (0, "HO5'") in atom_keys
    assert (0, "P") not in atom_keys
    assert (0, "HO3'") not in atom_keys
    assert (1, "P") in atom_keys
    assert (1, "OP3") not in atom_keys
    assert (1, "HO3'") not in atom_keys
    assert (7, "P") in atom_keys
    assert (7, "HO3'") in atom_keys
    assert (7, "OP3") not in atom_keys
    assert next(atom for atom in graph.atoms if atom.key == (0, "HO5'")).definition.ideal_coordinate is None


def test_compile_rna_polymer_sequence_handles_a_single_nucleotide_without_links():
    from mdna.chemistry import compile_rna_polymer_sequence

    graph = compile_rna_polymer_sequence("a")
    atom_keys = {atom.key for atom in graph.atoms}

    assert graph.component_ids == ("RNA:A",)
    assert graph.link_ids == ()
    assert all(bond[0] == bond[2] == 0 for bond in graph.bonds)
    assert graph.residue_state_ids == ("5prime_hydroxyl+3prime_hydroxyl",)
    assert (0, "HO5'") in atom_keys
    assert (0, "HO3'") in atom_keys


def test_compile_rna_polymer_sequence_rejects_unknown_symbols():
    from mdna.chemistry import compile_rna_polymer_sequence

    with pytest.raises(ValueError, match="Unsupported RNA nucleotide code"):
        compile_rna_polymer_sequence("AUZ")
