"""Registered identities for nucleotide templates bundled with MDNA.

This registry describes the current deoxyribose template set. It does not
declare RNA, arbitrary modified nucleotides, or XNA support.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re


@dataclass(frozen=True)
class AtomDefinition:
    """Chemical identity and optional ideal coordinate for one atom."""

    name: str
    element: str
    charge: int = 0
    ideal_coordinate: tuple[float, float, float] | None = None

    def __post_init__(self):
        if not self.name.strip():
            raise ValueError("Atom name must not be empty.")
        if not self.element.strip():
            raise ValueError(f"Atom {self.name!r} must declare an element.")
        if self.ideal_coordinate is not None:
            if len(self.ideal_coordinate) != 3 or not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in self.ideal_coordinate
            ):
                raise ValueError(
                    f"Atom {self.name!r} ideal coordinate must contain three finite values."
                )


@dataclass(frozen=True)
class BondDefinition:
    """An intracomponent covalent bond."""

    atom_1: str
    atom_2: str
    order: str

    def __post_init__(self):
        if not self.atom_1.strip() or not self.atom_2.strip():
            raise ValueError("Bond atom names must not be empty.")
        if self.atom_1 == self.atom_2:
            raise ValueError("A bond must connect two different atoms.")
        if not self.order.strip():
            raise ValueError("Bond order must not be empty.")


@dataclass(frozen=True)
class ComponentDefinition:
    """Source-traceable chemical graph for one polymer component."""

    component_id: str
    polymer_type: str
    sugar_chemistry: str
    backbone_chemistry: str
    residue_name: str
    atoms: tuple[AtomDefinition, ...]
    bonds: tuple[BondDefinition, ...]
    source: str
    sha256: str
    coordinate_unit: str | None = None
    template_key: str | None = None

    def __post_init__(self):
        for field_name in (
            "component_id",
            "polymer_type",
            "sugar_chemistry",
            "backbone_chemistry",
            "residue_name",
            "source",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty.")

        object.__setattr__(self, "atoms", tuple(self.atoms))
        object.__setattr__(self, "bonds", tuple(self.bonds))
        atom_names = [atom.name for atom in self.atoms]
        if len(atom_names) != len(set(atom_names)):
            raise ValueError(f"Component {self.component_id!r} has duplicate atom names.")
        atom_name_set = set(atom_names)
        for bond in self.bonds:
            missing = {bond.atom_1, bond.atom_2} - atom_name_set
            if missing:
                raise ValueError(
                    f"Component {self.component_id!r} bond endpoint(s) are absent: "
                    f"{', '.join(sorted(missing))}."
                )
        bond_pairs = [frozenset((bond.atom_1, bond.atom_2)) for bond in self.bonds]
        if len(bond_pairs) != len(set(bond_pairs)):
            raise ValueError(f"Component {self.component_id!r} has duplicate bonds.")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", self.sha256):
            raise ValueError("Component sha256 must contain exactly 64 hexadecimal characters.")
        if any(atom.ideal_coordinate is not None for atom in self.atoms):
            if not self.coordinate_unit:
                raise ValueError("coordinate_unit is required when atom coordinates are present.")


@dataclass(frozen=True)
class LinkDefinition:
    """Directed covalent connection between two component definitions."""

    link_id: str
    upstream_component_id: str
    downstream_component_id: str
    upstream_port: str
    downstream_port: str
    bond_order: str
    source: str
    evidence_sha256: str
    remove_upstream_atoms: tuple[str, ...] = ()
    remove_downstream_atoms: tuple[str, ...] = ()

    def __post_init__(self):
        for field_name in (
            "link_id",
            "upstream_component_id",
            "downstream_component_id",
            "upstream_port",
            "downstream_port",
            "bond_order",
            "source",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty.")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", self.evidence_sha256):
            raise ValueError("Link evidence_sha256 must contain exactly 64 hexadecimal characters.")
        object.__setattr__(self, "remove_upstream_atoms", tuple(self.remove_upstream_atoms))
        object.__setattr__(self, "remove_downstream_atoms", tuple(self.remove_downstream_atoms))


@dataclass(frozen=True)
class ComponentState:
    """A source-traceable atom graph for one RNA terminal/linkage state."""

    component_id: str
    state_id: str
    residue_name: str
    five_prime_state: str
    three_prime_state: str
    atoms: tuple[AtomDefinition, ...]
    bonds: tuple[BondDefinition, ...]
    source: str
    component_sha256: str
    evidence_sha256: str
    coordinate_unit: str
    coordinate_status: str

    def __post_init__(self):
        object.__setattr__(self, "atoms", tuple(self.atoms))
        object.__setattr__(self, "bonds", tuple(self.bonds))
        atom_names = [atom.name for atom in self.atoms]
        if len(atom_names) != len(set(atom_names)):
            raise ValueError(f"State {self.state_id!r} has duplicate atom names.")
        atom_name_set = set(atom_names)
        for bond in self.bonds:
            missing = {bond.atom_1, bond.atom_2} - atom_name_set
            if missing:
                raise ValueError(
                    f"State {self.state_id!r} bond endpoint(s) are absent: "
                    f"{', '.join(sorted(missing))}."
                )
        if self.coordinate_status not in {"complete_from_component", "incomplete"}:
            raise ValueError("Unsupported component-state coordinate status.")
        if not self.coordinate_unit.strip() or not self.source.strip():
            raise ValueError("Component-state source and coordinate unit are required.")
        for digest_name in ("component_sha256", "evidence_sha256"):
            if not re.fullmatch(r"[0-9a-fA-F]{64}", getattr(self, digest_name)):
                raise ValueError(f"{digest_name} must contain exactly 64 hexadecimal characters.")
        has_missing_coordinates = any(atom.ideal_coordinate is None for atom in self.atoms)
        expected_status = "incomplete" if has_missing_coordinates else "complete_from_component"
        if self.coordinate_status != expected_status:
            raise ValueError(
                f"coordinate_status must be {expected_status!r} for the state atom coordinates."
            )


@dataclass(frozen=True)
class PolymerAtom:
    """An atom retained at one residue position in a compiled polymer."""

    residue_index: int
    definition: AtomDefinition

    @property
    def name(self) -> str:
        return self.definition.name

    @property
    def key(self) -> tuple[int, str]:
        return self.residue_index, self.name


@dataclass(frozen=True)
class PolymerGraph:
    """Compiled atom-name graph for a directed linear polymer chain.

    Atoms retain their source ``AtomDefinition`` and expose a stable
    ``(residue_index, atom_name)`` key. Bond keys are
    ``(residue_1, atom_1, residue_2, atom_2, order)`` tuples so callers can
    preserve the component graph and inter-residue links without guessing
    polymer chemistry from residue names.
    """

    component_ids: tuple[str, ...]
    atoms: tuple[PolymerAtom, ...]
    bonds: tuple[tuple[int, str, int, str, str], ...]
    residue_states: tuple[ComponentState | None, ...]
    link_ids: tuple[str, ...]

    @property
    def residue_state_ids(self) -> tuple[str, ...]:
        return tuple(
            state.state_id if state is not None else "registered_component"
            for state in self.residue_states
        )


def compile_polymer_graph(
    component_ids: tuple[str, ...] | list[str],
    link_ids: tuple[str, ...] | list[str],
    registry: "ComponentRegistry | None" = None,
) -> PolymerGraph:
    """Compile component graphs and explicit directed links into a chain.

    ``link_ids[i]`` connects residue ``i`` to residue ``i + 1``. Atoms listed
    as leaving groups on that link are removed from the corresponding
    residues. A topology with terminal protonation or coordinates must be
    prepared separately; this function only compiles registered atom and bond
    identities.
    """

    registry = COMPONENT_REGISTRY if registry is None else registry
    if isinstance(component_ids, str) or isinstance(link_ids, str):
        raise TypeError("component_ids and link_ids must be ordered sequences, not strings.")
    component_ids = tuple(component_ids)
    link_ids = tuple(link_ids)
    expected_links = max(0, len(component_ids) - 1)
    if len(link_ids) != expected_links:
        raise ValueError(
            "A linear polymer requires one link per adjacent residue pair "
            f"({expected_links} expected, {len(link_ids)} provided)."
        )

    if registry is COMPONENT_REGISTRY:
        components = tuple(
            get_component_definition(component_id) for component_id in component_ids
        )
    else:
        components = tuple(registry.get(component_id) for component_id in component_ids)
    links = tuple(registry.get_link(link_id) for link_id in link_ids)
    active_atoms = [
        {atom.name: atom for atom in component.atoms} for component in components
    ]

    for index, link in enumerate(links):
        upstream = components[index]
        downstream = components[index + 1]
        if (
            link.upstream_component_id != upstream.component_id
            or link.downstream_component_id != downstream.component_id
        ):
            raise ValueError(
                f"Link {link.link_id!r} does not connect the ordered components "
                f"{upstream.component_id!r} -> {downstream.component_id!r}."
            )
        registry.validate_link(link)
        for atom_name in link.remove_upstream_atoms:
            active_atoms[index].pop(atom_name)
        for atom_name in link.remove_downstream_atoms:
            active_atoms[index + 1].pop(atom_name)

    atoms = tuple(
        PolymerAtom(residue_index, atom)
        for residue_index, residue_atoms in enumerate(active_atoms)
        for atom in residue_atoms.values()
    )
    bonds = []
    active_atom_keys = {atom.key for atom in atoms}
    for residue_index, component in enumerate(components):
        for bond in component.bonds:
            atom_1 = (residue_index, bond.atom_1)
            atom_2 = (residue_index, bond.atom_2)
            if atom_1 in active_atom_keys and atom_2 in active_atom_keys:
                bonds.append(
                    (residue_index, bond.atom_1, residue_index, bond.atom_2, bond.order)
                )
    for residue_index, link in enumerate(links):
        bonds.append(
            (
                residue_index,
                link.upstream_port,
                residue_index + 1,
                link.downstream_port,
                link.bond_order,
            )
        )
    return PolymerGraph(
        component_ids=component_ids,
        atoms=atoms,
        bonds=tuple(bonds),
        residue_states=tuple(None for _ in components),
        link_ids=link_ids,
    )


class ComponentRegistry:
    """In-memory registry for explicitly identified chemical components."""

    def __init__(self, components=()):
        self._components = {}
        self._links = {}
        for component in components:
            self.register(component)

    def register(self, component: ComponentDefinition) -> None:
        if not isinstance(component, ComponentDefinition):
            raise TypeError("Only ComponentDefinition instances can be registered.")
        if component.component_id in self._components:
            raise ValueError(f"Component {component.component_id!r} is already registered.")
        self._components[component.component_id] = component

    def get(self, component_id: str) -> ComponentDefinition:
        try:
            return self._components[component_id]
        except KeyError as error:
            raise ValueError(f"Unknown component {component_id!r}.") from error

    def validate_link(self, link: LinkDefinition) -> None:
        if not isinstance(link, LinkDefinition):
            raise TypeError("link must be a LinkDefinition instance.")
        upstream = self.get(link.upstream_component_id)
        downstream = self.get(link.downstream_component_id)
        upstream_atoms = {atom.name for atom in upstream.atoms}
        downstream_atoms = {atom.name for atom in downstream.atoms}

        if link.upstream_port not in upstream_atoms:
            raise ValueError(
                f"Link {link.link_id!r} upstream port {link.upstream_port!r} "
                f"is absent from {upstream.component_id!r}."
            )
        if link.downstream_port not in downstream_atoms:
            raise ValueError(
                f"Link {link.link_id!r} downstream port {link.downstream_port!r} "
                f"is absent from {downstream.component_id!r}."
            )
        if link.upstream_port in link.remove_upstream_atoms:
            raise ValueError("A link cannot remove its upstream port atom.")
        if link.downstream_port in link.remove_downstream_atoms:
            raise ValueError("A link cannot remove its downstream port atom.")
        missing_upstream = set(link.remove_upstream_atoms) - upstream_atoms
        if missing_upstream:
            raise ValueError(
                f"Link {link.link_id!r} removes absent upstream atom(s): "
                f"{', '.join(sorted(missing_upstream))}."
            )
        missing_downstream = set(link.remove_downstream_atoms) - downstream_atoms
        if missing_downstream:
            raise ValueError(
                f"Link {link.link_id!r} removes absent downstream atom(s): "
                f"{', '.join(sorted(missing_downstream))}."
            )

    def register_link(self, link: LinkDefinition) -> None:
        self.validate_link(link)
        if link.link_id in self._links:
            raise ValueError(f"Link {link.link_id!r} is already registered.")
        self._links[link.link_id] = link

    def get_link(self, link_id: str) -> LinkDefinition:
        try:
            return self._links[link_id]
        except KeyError as error:
            raise ValueError(f"Unknown link {link_id!r}.") from error


def _load_packaged_rna_components() -> tuple[ComponentDefinition, ...]:
    """Load audited offline CCD records bundled with the Python package."""

    data_path = Path(__file__).parent / "data" / "rna_components.json"
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported RNA component data schema in {data_path}.")
    components = []
    for record in payload["components"]:
        atoms = tuple(
            AtomDefinition(
                name=atom["name"],
                element=atom["element"],
                charge=int(atom["charge"]),
                ideal_coordinate=tuple(atom["ideal_coordinate"]),
            )
            for atom in record["atoms"]
        )
        bonds = tuple(
            BondDefinition(
                atom_1=bond["atom_1"],
                atom_2=bond["atom_2"],
                order=bond["order"],
            )
            for bond in record["bonds"]
        )
        components.append(
            ComponentDefinition(
                component_id=record["component_id"],
                polymer_type=record["polymer_type"],
                sugar_chemistry=record["sugar_chemistry"],
                backbone_chemistry=record["backbone_chemistry"],
                residue_name=record["residue_name"],
                atoms=atoms,
                bonds=bonds,
                source=record["source"],
                sha256=record["sha256"],
                coordinate_unit=record["coordinate_unit"],
                template_key=record["template_key"],
            )
        )
    return tuple(components)


COMPONENT_REGISTRY = ComponentRegistry(_load_packaged_rna_components())


def get_component_definition(component_id: str) -> ComponentDefinition:
    """Return a registered chemical component by its explicit component ID."""
    try:
        return COMPONENT_REGISTRY.get(component_id)
    except ValueError:
        if not component_id.startswith("DNA:"):
            raise

    legacy_component_id = component_id.split(":", 1)[1]
    matches = [
        definition
        for definition in NUCLEOTIDE_REGISTRY.values()
        if definition.component_id == legacy_component_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Unknown component {component_id!r}.")
    component = _load_legacy_component(matches[0])
    COMPONENT_REGISTRY.register(component)
    return component


def derive_rna_component_state(
    component_id: str,
    *,
    five_prime: str,
    three_prime: str,
) -> ComponentState:
    """Derive an RNA residue atom graph from explicit end/linkage states.

    ``five_prime`` accepts ``"hydroxyl"``, ``"phosphodiester"`` or
    ``"monophosphate"``; ``three_prime`` accepts ``"hydroxyl"`` or
    ``"phosphodiester"``. Coordinates remain CCD ideal coordinates except for
    the newly added 5-prime hydroxyl hydrogen, whose coordinate is left unset.
    The state is therefore chemically/topologically prepared but not always
    ready for coordinate output.
    """

    allowed_five_prime_states = {"hydroxyl", "phosphodiester", "monophosphate"}
    allowed_three_prime_states = {"hydroxyl", "phosphodiester"}
    if five_prime not in allowed_five_prime_states:
        raise ValueError(f"Unsupported RNA 5-prime state {five_prime!r}.")
    if three_prime not in allowed_three_prime_states:
        raise ValueError(f"Unsupported RNA 3-prime state {three_prime!r}.")
    component = get_component_definition(component_id)
    if component.polymer_type != "RNA":
        raise ValueError(
            f"RNA end-state derivation requires an RNA component; got {component_id!r}."
        )

    atoms = {atom.name: atom for atom in component.atoms}
    bonds = list(component.bonds)
    if five_prime == "hydroxyl":
        for atom_name in ("P", "OP1", "OP2", "OP3", "HOP2", "HOP3"):
            atoms.pop(atom_name, None)
        if "O5'" not in atoms:
            raise ValueError(f"Component {component_id!r} lacks the O5' attachment atom.")
        atoms["HO5'"] = AtomDefinition(name="HO5'", element="H")
        bonds.append(BondDefinition("O5'", "HO5'", "single"))
    elif five_prime == "phosphodiester":
        for atom_name in ("OP3", "HOP2", "HOP3"):
            atoms.pop(atom_name, None)

    if three_prime == "phosphodiester":
        atoms.pop("HO3'", None)

    atom_names = set(atoms)
    bonds = [
        bond
        for bond in bonds
        if bond.atom_1 in atom_names and bond.atom_2 in atom_names
    ]
    state_id = f"5prime_{five_prime}+3prime_{three_prime}"
    evidence_hash = _RNA_LINK_EVIDENCE_SHA256
    coordinate_status = (
        "complete_from_component"
        if all(atom.ideal_coordinate is not None for atom in atoms.values())
        else "incomplete"
    )
    return ComponentState(
        component_id=component.component_id,
        state_id=state_id,
        residue_name=component.residue_name,
        five_prime_state=five_prime,
        three_prime_state=three_prime,
        atoms=tuple(atoms.values()),
        bonds=tuple(bonds),
        source=(
            f"{component.source}; terminal-state observation: "
            "artifacts/rna_pilot/reference/4U3R.audit.json"
        ),
        component_sha256=component.sha256,
        evidence_sha256=evidence_hash,
        coordinate_unit=component.coordinate_unit or "angstrom",
        coordinate_status=coordinate_status,
    )


def get_nucleotide_component(code: str) -> ComponentDefinition:
    """Resolve a legacy sequence code to its explicit DNA template component."""

    definition = get_nucleotide_definition(code)
    return get_component_definition(f"DNA:{definition.component_id}")


def get_link_definition(link_id: str) -> LinkDefinition:
    """Return a registered directed connection rule by its explicit ID."""

    return COMPONENT_REGISTRY.get_link(link_id)


def compile_rna_polymer_sequence(sequence: str) -> PolymerGraph:
    """Compile a standard A/C/G/U RNA sequence into a directed chain graph.

    The first residue is 5-prime hydroxyl terminated, the last residue is
    3-prime hydroxyl terminated, and intervening termini use the registered
    phosphodiester links. The returned atoms retain component-local CCD ideal
    coordinates; this does not place or stitch residues in 3D space.
    """

    if not isinstance(sequence, str):
        raise TypeError("RNA sequence must be a string.")
    normalized = sequence.upper()
    if not normalized:
        raise ValueError("RNA sequence must not be empty.")
    unsupported = sorted(set(normalized) - set("ACGU"))
    if unsupported:
        raise ValueError(
            "Unsupported RNA nucleotide code(s): " + ", ".join(unsupported) + "."
        )

    component_ids = tuple(f"RNA:{base}" for base in normalized)
    residue_states = tuple(
        derive_rna_component_state(
            component_id,
            five_prime="hydroxyl" if index == 0 else "phosphodiester",
            three_prime=(
                "hydroxyl" if index == len(normalized) - 1 else "phosphodiester"
            ),
        )
        for index, component_id in enumerate(component_ids)
    )
    atoms = tuple(
        PolymerAtom(residue_index, atom)
        for residue_index, state in enumerate(residue_states)
        for atom in state.atoms
    )
    bonds = [
        (residue_index, bond.atom_1, residue_index, bond.atom_2, bond.order)
        for residue_index, state in enumerate(residue_states)
        for bond in state.bonds
    ]
    link_ids = tuple(
        f"{component_ids[index]}->{component_ids[index + 1]}:3to5"
        for index in range(len(component_ids) - 1)
    )
    for index, link_id in enumerate(link_ids):
        link = get_link_definition(link_id)
        if (
            link.upstream_component_id != component_ids[index]
            or link.downstream_component_id != component_ids[index + 1]
        ):
            raise ValueError(f"RNA link {link_id!r} does not match the sequence order.")
        upstream_atom_names = {atom.name for atom in residue_states[index].atoms}
        downstream_atom_names = {atom.name for atom in residue_states[index + 1].atoms}
        if link.upstream_port not in upstream_atom_names:
            raise ValueError(
                f"RNA state {residue_states[index].state_id!r} lacks link port "
                f"{link.upstream_port!r}."
            )
        if link.downstream_port not in downstream_atom_names:
            raise ValueError(
                f"RNA state {residue_states[index + 1].state_id!r} lacks link port "
                f"{link.downstream_port!r}."
            )
        bonds.append(
            (
                index,
                link.upstream_port,
                index + 1,
                link.downstream_port,
                link.bond_order,
            )
        )
    return PolymerGraph(
        component_ids=component_ids,
        atoms=atoms,
        bonds=tuple(bonds),
        residue_states=residue_states,
        link_ids=link_ids,
    )


def _load_legacy_component(definition: "NucleotideDefinition") -> ComponentDefinition:
    """Adapt one bundled MDTraj template to the explicit component contract."""

    import mdtraj as md

    from .utils import get_data_file_path

    relative_path = template_relative_path(definition.code)
    template_path = Path(get_data_file_path(relative_path))
    trajectory = md.load_hdf5(str(template_path))
    residues = list(trajectory.topology.residues)
    if len(residues) != 1:
        raise ValueError(
            f"Template for {definition.code!r} must contain exactly one residue; "
            f"found {len(residues)}."
        )
    residue = residues[0]
    if residue.name.strip().upper() != definition.template_residue_name:
        raise ValueError(
            f"Template for {definition.code!r} is registered as "
            f"{definition.template_residue_name!r}, but contains {residue.name!r}."
        )
    atoms = tuple(
        AtomDefinition(
            name=atom.name,
            element=atom.element.symbol if atom.element is not None else "unknown",
            ideal_coordinate=tuple(float(value) for value in trajectory.xyz[0, atom.index]),
        )
        for atom in residue.atoms
    )
    bonds = tuple(
        BondDefinition(
            atom_1=atom_1.name,
            atom_2=atom_2.name,
            order=str(getattr(bond, "order", None) or "unspecified"),
        )
        for bond in trajectory.topology.bonds
        for atom_1, atom_2 in (bond,)
    )
    digest = hashlib.sha256(template_path.read_bytes()).hexdigest()
    return ComponentDefinition(
        component_id=f"DNA:{definition.component_id}",
        polymer_type="DNA",
        sugar_chemistry=definition.sugar_chemistry,
        backbone_chemistry=definition.backbone_chemistry,
        residue_name=definition.residue_name,
        atoms=atoms,
        bonds=bonds,
        source=(
            f"{definition.template_source}; bundled template {relative_path}; "
            f"component identity: {definition.component_identity_source}"
        ),
        sha256=digest,
        coordinate_unit="nm",
        template_key=relative_path,
    )


@dataclass(frozen=True)
class NucleotideDefinition:
    """Chemical identity and template mapping for one sequence code."""

    code: str
    template_key: str
    template_residue_name: str
    residue_name: str
    component_id: str
    complement: str
    sugar_chemistry: str = "2-deoxyribose"
    backbone_chemistry: str = "phosphodiester"
    template_source: str = "bundled HDF5; original coordinate provenance unresolved"
    component_identity_source: str = "bundled topology label; CCD match pending"


# ``template_key`` names the bundled file BDNA_<template_key>.h5. The
# Hachimoji files are permuted: the sequence letter must not select its
# same-named file. Component IDs and residue names below were read from the
# bundled template topologies. Hachimoji identities were cross-checked against
# wwPDB CCD records; coordinate provenance remains unresolved.
NUCLEOTIDE_REGISTRY = {
    "A": NucleotideDefinition("A", "A", "DA", "DA", "DA", "T"),
    "C": NucleotideDefinition("C", "C", "DC", "DC", "DC", "G"),
    "G": NucleotideDefinition("G", "G", "DG", "DG", "DG", "C"),
    "T": NucleotideDefinition("T", "T", "DT", "DT", "DT", "A"),
    # The bundled U template lacks the sugar O2' atom. It represents a
    # deoxyuridine-like component and must not be used as an RNA template.
    "U": NucleotideDefinition("U", "U", "U", "DU", "DU", "A"),
    "D": NucleotideDefinition("D", "D", "TC1", "TC1", "TC1", "G"),
    "E": NucleotideDefinition("E", "E", "2PR", "2PR", "2PR", "T"),
    "L": NucleotideDefinition("L", "L", "LHO", "LHO", "LHO", "M"),
    "M": NucleotideDefinition("M", "M", "BMN", "BMN", "BMN", "L"),
    "B": NucleotideDefinition(
        "B", "P", "IGU", "IGU", "IGU", "S",
        component_identity_source="RCSB CCD record cross-checked",
    ),
    "S": NucleotideDefinition(
        "S", "Z", "JSP", "JSP", "JSP", "B",
        component_identity_source="RCSB CCD record cross-checked",
    ),
    "Z": NucleotideDefinition(
        "Z", "S", "1W5", "1W5", "1W5", "P",
        component_identity_source="RCSB CCD record cross-checked",
    ),
    "P": NucleotideDefinition(
        "P", "B", "1WA", "1WA", "1WA", "Z",
        component_identity_source="RCSB CCD record cross-checked",
    ),
}

# This is the default partner selected when a code is supplied on the first
# strand. It is intentionally directional: A defaults to T, while a U input
# defaults to A. A complete pairing-compatibility matrix is future work.
NUCLEOTIDE_COMPLEMENTS = {
    code: definition.complement
    for code, definition in NUCLEOTIDE_REGISTRY.items()
}


_RNA_LINK_EVIDENCE = (
    "wwPDB polymer linkage definition and observed internal RNA residues in "
    "artifacts/rna_pilot/reference/4U3R.audit.json"
)
_RNA_LINK_EVIDENCE_SHA256 = "2e4eb14d82021b925e1aafef4263842f09ba35489c58d07f5978ff0a8c4eb105"
for _upstream in "ACGU":
    for _downstream in "ACGU":
        _link = LinkDefinition(
            link_id=f"RNA:{_upstream}->RNA:{_downstream}:3to5",
            upstream_component_id=f"RNA:{_upstream}",
            downstream_component_id=f"RNA:{_downstream}",
            upstream_port="O3'",
            downstream_port="P",
            bond_order="single",
            source=_RNA_LINK_EVIDENCE,
            evidence_sha256=_RNA_LINK_EVIDENCE_SHA256,
            remove_upstream_atoms=("HO3'",),
            remove_downstream_atoms=("OP3", "HOP3"),
        )
        COMPONENT_REGISTRY.register_link(_link)


def get_nucleotide_definition(code: str) -> NucleotideDefinition:
    """Return a registered definition or raise a useful input error."""

    normalized = str(code).upper()
    try:
        return NUCLEOTIDE_REGISTRY[normalized]
    except KeyError as error:
        supported = ", ".join(NUCLEOTIDE_REGISTRY)
        raise ValueError(
            f"Unsupported nucleotide code {code!r}. Registered codes: {supported}."
        ) from error


def template_relative_path(code: str) -> str:
    """Return the bundled MDTraj HDF5 path for a sequence code."""

    definition = get_nucleotide_definition(code)
    return f"atomic/bases/BDNA_{definition.template_key}.h5"


def copy_template_bonds(topology, template_topology, atom_map, require_all=False):
    """Copy mapped bonds from a template topology into ``topology``.

    ``atom_map`` maps template MDTraj atoms to destination MDTraj atoms. If
    ``require_all`` is false, bonds with an unmapped endpoint are skipped; this
    allows mutation code to retain the target sugar/backbone while replacing
    only base atoms.
    """

    copied = 0
    skipped = []
    existing = {
        frozenset((atom_1.index, atom_2.index))
        for atom_1, atom_2 in topology.bonds
    }

    for bond in template_topology.bonds:
        atom_1, atom_2 = bond
        if atom_1 not in atom_map or atom_2 not in atom_map:
            skipped.append(bond)
            continue

        target_1, target_2 = atom_map[atom_1], atom_map[atom_2]
        pair = frozenset((target_1.index, target_2.index))
        if pair in existing:
            continue

        kwargs = {}
        bond_type = getattr(bond, "type", None)
        bond_order = getattr(bond, "order", None)
        if bond_type is not None:
            kwargs["type"] = bond_type
        if bond_order is not None:
            kwargs["order"] = bond_order
        topology.add_bond(target_1, target_2, **kwargs)
        existing.add(pair)
        copied += 1

    if require_all and skipped:
        raise ValueError(
            f"Could not map {len(skipped)} of {len(template_topology.bonds)} "
            "template bonds onto the destination topology."
        )
    return copied
