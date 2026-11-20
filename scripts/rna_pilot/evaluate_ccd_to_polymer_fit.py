"""Evaluate rigid placement of CCD RNA monomers against polymer coordinates.

This diagnostic fits each CCD ideal component to the observed base ring of
each residue in the archived RNA ion series. It then measures the remaining
heavy-atom deviations and the O3'-to-next-P distance implied by independently
placed monomers. It does not construct or validate an MDNA RNA model.

Run in the project WSL ``dip2`` environment::

    python scripts/rna_pilot/evaluate_ccd_to_polymer_fit.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ENTRY_IDS = ("4u3l", "4u3o", "4u3p", "4u3r", "4u47", "4u78")
PURINE_RING = ("N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4")
PYRIMIDINE_RING = ("N1", "C2", "N3", "C4", "C5", "C6")
BASE_RING = {"A": PURINE_RING, "G": PURINE_RING, "C": PYRIMIDINE_RING, "U": PYRIMIDINE_RING}
RNA_NAMES = set(BASE_RING)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cif_string(value: str) -> str:
    import gemmi

    return gemmi.cif.as_string(value).strip()


def read_ccd_component(path: Path) -> tuple[str, dict[str, np.ndarray], dict[str, str]]:
    import gemmi

    block = gemmi.cif.read_file(str(path)).sole_block()
    component_id = _cif_string(block.find_value("_chem_comp.id"))
    rows = block.find(
        [
            "_chem_comp_atom.atom_id",
            "_chem_comp_atom.type_symbol",
            "_chem_comp_atom.pdbx_model_Cartn_x_ideal",
            "_chem_comp_atom.pdbx_model_Cartn_y_ideal",
            "_chem_comp_atom.pdbx_model_Cartn_z_ideal",
        ]
    )
    coordinates: dict[str, np.ndarray] = {}
    elements: dict[str, str] = {}
    for row in rows:
        atom_name = _cif_string(row[0])
        values = [float(row[index]) for index in (2, 3, 4)]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path.name}: non-finite ideal coordinates for {atom_name}")
        coordinates[atom_name] = np.asarray(values, dtype=float)
        elements[atom_name] = _cif_string(row[1]).upper()
    return component_id, coordinates, elements


def rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return row-vector rotation and translation mapping source onto target."""

    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must have the same shape (n_atoms, 3)")
    if source.shape[0] < 3:
        raise ValueError("at least three base-ring atoms are required for a rigid fit")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("source and target coordinates must be finite")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, singular_values, right = np.linalg.svd(covariance)
    if singular_values[1] <= np.finfo(float).eps * singular_values[0]:
        raise ValueError("base-ring coordinates do not define a unique rigid orientation")
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    translation = target_center - source_center @ rotation
    return rotation, translation


def residue_atoms(residue: gemmi.Residue) -> dict[str, gemmi.Atom]:
    result: dict[str, gemmi.Atom] = {}
    for atom in residue:
        name = atom.name.strip()
        previous = result.get(name)
        if previous is None or atom.occ > previous.occ:
            result[name] = atom
    return result


def xyz(atom: gemmi.Atom) -> np.ndarray:
    return np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)


def audit_entry(
    entry_id: str,
    structure_path: Path,
    components: dict[str, tuple[dict[str, np.ndarray], dict[str, str]]],
) -> dict[str, Any]:
    import gemmi

    structure = gemmi.read_structure(str(structure_path))
    if len(structure) != 1:
        raise ValueError(f"{entry_id}: expected one model, found {len(structure)}")
    chain_results = []
    for chain in structure[0]:
        residues = [residue for residue in chain if residue.name.strip().upper() in RNA_NAMES]
        if not residues:
            continue
        fitted = []
        for residue in residues:
            component_id = residue.name.strip().upper()
            ideal, elements = components[component_id]
            observed = residue_atoms(residue)
            ring_atoms = BASE_RING[component_id]
            missing_ring = sorted(
                (set(ring_atoms) - ideal.keys()) | (set(ring_atoms) - observed.keys())
            )
            if missing_ring:
                raise ValueError(
                    f"{entry_id} chain {chain.name} {residue.seqid}: "
                    f"missing ring atoms {missing_ring}"
                )
            source_ring = np.stack([ideal[name] for name in ring_atoms])
            target_ring = np.stack([xyz(observed[name]) for name in ring_atoms])
            rotation, translation = rigid_fit(source_ring, target_ring)
            ring_fitted = source_ring @ rotation + translation
            ring_rmsd = float(np.sqrt(np.mean(np.sum((ring_fitted - target_ring) ** 2, axis=1))))
            common_heavy = sorted(
                name
                for name in ideal.keys() & observed.keys()
                if elements[name] != "H" and observed[name].element.name.upper() != "H"
            )
            transformed = {name: coord @ rotation + translation for name, coord in ideal.items()}
            atom_errors = {
                name: float(np.linalg.norm(transformed[name] - xyz(observed[name])))
                for name in common_heavy
            }
            sugar_names = [name for name in common_heavy if name in {"C1'", "C2'", "O2'", "C3'", "O3'", "C4'", "O4'", "C5'", "O5'"}]
            backbone_names = [name for name in common_heavy if name in {"P", "OP1", "OP2", "OP3", "O5'", "C5'", "C4'", "C3'", "O3'"}]
            fitted.append(
                {
                    "residue": f"{component_id}:{residue.seqid}",
                    "component_id": component_id,
                    "atom_count_common_heavy": len(common_heavy),
                    "base_ring_fit_rmsd_A": ring_rmsd,
                    "all_heavy_rmsd_after_base_fit_A": float(
                        math.sqrt(sum(atom_errors[name] ** 2 for name in common_heavy) / len(common_heavy))
                    ),
                    "sugar_rmsd_after_base_fit_A": float(
                        math.sqrt(sum(atom_errors[name] ** 2 for name in sugar_names) / len(sugar_names))
                    ),
                    "backbone_rmsd_after_base_fit_A": float(
                        math.sqrt(sum(atom_errors[name] ** 2 for name in backbone_names) / len(backbone_names))
                    ),
                    "transformed_O3_prime_A": transformed["O3'"].tolist(),
                    "transformed_P_A": transformed["P"].tolist(),
                    "observed_O3_prime_A": xyz(observed["O3'"]).tolist() if "O3'" in observed else None,
                    "observed_P_A": xyz(observed["P"]).tolist() if "P" in observed else None,
                }
            )

        links = []
        for upstream, downstream in zip(fitted, fitted[1:]):
            predicted = float(
                np.linalg.norm(
                    np.asarray(upstream["transformed_O3_prime_A"])
                    - np.asarray(downstream["transformed_P_A"])
                )
            )
            observed = None
            if upstream["observed_O3_prime_A"] is not None and downstream["observed_P_A"] is not None:
                observed = float(
                    np.linalg.norm(
                        np.asarray(upstream["observed_O3_prime_A"])
                        - np.asarray(downstream["observed_P_A"])
                    )
                )
            links.append(
                {
                    "upstream": upstream["residue"],
                    "downstream": downstream["residue"],
                    "independently_fitted_O3_prime_to_next_P_A": predicted,
                    "observed_O3_prime_to_next_P_A": observed,
                }
            )
        chain_results.append(
            {
                "chain_id": chain.name,
                "residue_count": len(fitted),
                "residues": [
                    {
                        key: value
                        for key, value in record.items()
                        if not key.endswith("_A") or "rmsd" in key.lower()
                    }
                    for record in fitted
                ],
                "links": links,
            }
        )
    return {
        "entry_id": entry_id.upper(),
        "source_coordinate_file": structure_path.name,
        "source_coordinate_sha256": sha256(structure_path),
        "chains": chain_results,
    }


def summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    metric_values: dict[str, list[float]] = {
        "base_ring_fit_rmsd_A": [],
        "all_heavy_rmsd_after_base_fit_A": [],
        "sugar_rmsd_after_base_fit_A": [],
        "backbone_rmsd_after_base_fit_A": [],
        "independently_fitted_O3_prime_to_next_P_A": [],
        "observed_O3_prime_to_next_P_A": [],
    }
    residue_count = link_count = 0
    for entry in entries:
        for chain in entry["chains"]:
            residue_count += chain["residue_count"]
            for residue in chain["residues"]:
                for key in metric_values:
                    if key in residue:
                        metric_values[key].append(residue[key])
            for link in chain["links"]:
                link_count += 1
                for key in metric_values:
                    if key in link and link[key] is not None:
                        metric_values[key].append(link[key])
    summaries = {}
    for name, values in metric_values.items():
        if values:
            array = np.asarray(values, dtype=float)
            summaries[name] = {
                "count": len(values),
                "mean": float(array.mean()),
                "median": float(np.median(array)),
                "p95": float(np.percentile(array, 95)),
                "min": float(array.min()),
                "max": float(array.max()),
            }
    return {
        "entry_count": len(entries),
        "residue_count": residue_count,
        "adjacent_link_count": link_count,
        "metrics": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=project_root / "artifacts/rna_pilot/reference/ion_series",
    )
    parser.add_argument(
        "--ccd-dir",
        type=Path,
        default=project_root / "artifacts/rna_pilot/reference/ccd",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "artifacts/rna_pilot/reference/ccd_to_polymer_fit.json",
    )
    args = parser.parse_args()
    ccd_dir = args.ccd_dir.resolve()
    components = {}
    ccd_sources = []
    for code in "ACGU":
        path = ccd_dir / f"{code}.cif"
        component_id, coordinates, elements = read_ccd_component(path)
        if component_id != code:
            raise ValueError(f"{path.name}: expected CCD component {code}, found {component_id}")
        components[code] = (coordinates, elements)
        ccd_sources.append({"file": path.name, "sha256": sha256(path)})
    entries = [
        audit_entry(entry_id, args.reference_dir.resolve() / f"{entry_id}.cif", components)
        for entry_id in ENTRY_IDS
    ]
    report = {
        "audit_tool": "scripts/rna_pilot/evaluate_ccd_to_polymer_fit.py",
        "coordinate_unit": "angstrom",
        "base_ring_atoms": {key: list(value) for key, value in BASE_RING.items()},
        "fit_method": "proper-rotation Kabsch fit of CCD ideal heavy-atom base ring to each observed residue base ring",
        "input_ccd_components": ccd_sources,
        "summary": summarize(entries),
        "entries": entries,
        "interpretation_limit": (
            "This audit tests independent rigid placement of ideal CCD monomers against observed polymer residues. "
            "It does not optimize torsions, join coordinates, generate a new RNA model, evaluate density, or establish universal support."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
