"""Compute simple, explicit base-pair-frame geometry for the RNA ion series.

The reported rise and twist are descriptive project metrics. They are not
standard 3DNA/CEHS base-step parameters and must not be compared to those
definitions without an explicit coordinate/method crosswalk.

Run with the project WSL ``dip2`` environment::

    python scripts/rna_pilot/compare_ion_series_geometry.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gemmi
import numpy as np


ENTRY_IDS = ("4u3l", "4u3o", "4u3p", "4u3r", "4u47", "4u78")
RING_ATOMS = {
    "A": ("C2", "C4", "C5", "C6", "N1", "N3", "C8", "N7", "N9"),
    "G": ("C2", "C4", "C5", "C6", "N1", "N3", "C8", "N7", "N9"),
    "C": ("C2", "C4", "C5", "C6", "N1", "N3"),
    "U": ("C2", "C4", "C5", "C6", "N1", "N3"),
}
COMPLEMENT = {"A": "U", "U": "A", "C": "G", "G": "C"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atom_positions(residue) -> dict[str, np.ndarray]:
    selected = {}
    for atom in residue:
        name = atom.name.strip()
        altloc = atom.altloc in ("", " ", "\x00")
        old = selected.get(name)
        if old is None or (atom.occ, altloc) > (old[0], old[1]):
            selected[name] = (atom.occ, altloc, np.array([atom.pos.x, atom.pos.y, atom.pos.z]))
    return {name: item[2] for name, item in selected.items()}


def _base_frame(residue) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = residue.name.strip().upper()
    positions = _atom_positions(residue)
    names = RING_ATOMS[base]
    missing = sorted(set(names) - set(positions))
    if missing:
        raise ValueError(f"{base} {residue.seqid}: missing ring atoms {missing}")
    coordinates = np.array([positions[name] for name in names])
    center = coordinates.mean(axis=0)
    _, _, vh = np.linalg.svd(coordinates - center, full_matrices=False)
    normal = vh[-1] / np.linalg.norm(vh[-1])
    glycosidic_name = "N9" if base in ("A", "G") else "N1"
    glycosidic = positions[glycosidic_name]
    c1_prime = positions.get("C1'")
    if c1_prime is None:
        raise ValueError(f"{base} {residue.seqid}: missing C1'")
    glycosidic_direction = glycosidic - c1_prime
    return center, normal, glycosidic_direction / np.linalg.norm(glycosidic_direction)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector")
    return vector / norm


def _angle_about_axis(first: np.ndarray, second: np.ndarray, axis: np.ndarray) -> float:
    first = _unit(first - np.dot(first, axis) * axis)
    second = _unit(second - np.dot(second, axis) * axis)
    return float(np.degrees(np.arctan2(np.dot(axis, np.cross(first, second)), np.dot(first, second))))


def audit_entry(path: Path) -> dict[str, Any]:
    structure = gemmi.read_structure(str(path))
    if len(structure) != 1:
        raise ValueError(f"{path.name}: expected one model, found {len(structure)}")
    chains = []
    for chain in structure[0]:
        residues = [r for r in chain if r.name.strip().upper() in RING_ATOMS]
        if residues:
            chains.append((chain.name, residues))
    if len(chains) != 2 or any(len(residues) != 8 for _, residues in chains):
        raise ValueError(f"{path.name}: expected two 8-nt RNA chains")

    (chain_a_id, chain_a), (chain_b_id, chain_b) = chains
    chain_b_anti = list(reversed(chain_b))
    base_centers_a = np.array([_base_frame(residue)[0] for residue in chain_a])
    base_centers_b = np.array([_base_frame(residue)[0] for residue in chain_b])
    base_center_distances = np.linalg.norm(
        base_centers_a[:, np.newaxis, :] - base_centers_b[np.newaxis, :, :], axis=2
    )
    nearest_complement_indices = []
    for index, residue_a in enumerate(chain_a):
        required_base = COMPLEMENT[residue_a.name.strip().upper()]
        possible = [
            j for j, residue_b in enumerate(chain_b)
            if residue_b.name.strip().upper() == required_base
        ]
        nearest_complement_indices.append(
            min(possible, key=lambda j: base_center_distances[index, j])
        )
    pairs = []
    for index, (res_a, res_b) in enumerate(zip(chain_a, chain_b_anti), start=1):
        base_a = res_a.name.strip().upper()
        base_b = res_b.name.strip().upper()
        if COMPLEMENT[base_a] != base_b:
            raise ValueError(
                f"{path.name}: noncanonical pair at step {index}: {base_a}-{base_b}"
            )
        center_a, normal_a, glyco_a = _base_frame(res_a)
        center_b, normal_b, glyco_b = _base_frame(res_b)
        pairs.append(
            {
                "index": index,
                "pair": f"{base_a}-{base_b}",
                "center": (center_a + center_b) / 2,
                "centroid_a": center_a,
                "centroid_b": center_b,
                "normal_a": normal_a,
                "normal_b": normal_b,
                "glycosidic_a": glyco_a,
                "glycosidic_b": glyco_b,
                "pair_width": float(np.linalg.norm(center_a - center_b)),
            }
        )

    centers = np.array([pair["center"] for pair in pairs])
    centered = centers - centers.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = _unit(vh[0])
    if np.dot(axis, centers[-1] - centers[0]) < 0:
        axis = -axis

    segment_vectors = np.diff(centers, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    segment_axes = segment_vectors / segment_lengths[:, np.newaxis]
    global_axis_projections = segment_vectors @ axis
    bend_angles = np.degrees(
        np.arccos(np.clip(np.sum(segment_axes[:-1] * segment_axes[1:], axis=1), -1, 1))
    )
    local_tangents = []
    for index in range(len(centers)):
        if index == 0:
            tangent = segment_axes[0]
        elif index == len(centers) - 1:
            tangent = segment_axes[-1]
        else:
            tangent = _unit(segment_axes[index - 1] + segment_axes[index])
        local_tangents.append(tangent)
    radial_vectors = []
    plane_global_axis_angles = []
    plane_local_tangent_angles = []
    pair_widths = []
    for pair in pairs:
        radial = pair["centroid_a"] - pair["center"]
        radial -= np.dot(radial, axis) * axis
        radial_vectors.append(_unit(radial))
        normal_a = pair["normal_a"]
        normal_b = pair["normal_b"]
        if np.dot(normal_a, normal_b) < 0:
            normal_b = -normal_b
        combined_normal = _unit(normal_a + normal_b)
        plane_global_axis_angles.append(
            float(np.degrees(np.arccos(np.clip(abs(np.dot(combined_normal, axis)), -1, 1))))
        )
        plane_local_tangent_angles.append(
            float(np.degrees(np.arccos(np.clip(abs(np.dot(combined_normal, local_tangents[pair["index"] - 1])), -1, 1))))
        )
        pair_widths.append(pair["pair_width"])
    twists = [
        _angle_about_axis(first, second, axis)
        for first, second in zip(radial_vectors, radial_vectors[1:])
    ]
    local_twists = [
        _angle_about_axis(first, second, step_axis)
        for first, second, step_axis in zip(radial_vectors, radial_vectors[1:], segment_axes)
    ]

    return {
        "entry_id": path.stem.upper(),
        "coordinate_file": path.name,
        "coordinate_sha256": _sha256(path),
        "chain_ids": [chain_a_id, chain_b_id],
        "paired_sequence": [pair["pair"] for pair in pairs],
        "expected_reverse_pairing_partner_indices_zero_based": list(range(7, -1, -1)),
        "nearest_complementary_partner_indices_zero_based": nearest_complement_indices,
        "coordinate_nearest_partner_matches_expected_antiparallel": nearest_complement_indices == list(range(7, -1, -1)),
        "pair_centroid_distance_matrix_A": base_center_distances.tolist(),
        "axis_definition": "first principal component of the eight base-pair centers, oriented along chain A",
        "radial_definition": "chain A base-ring centroid projected onto the plane perpendicular to the global axis",
        "center_step_distance_definition": "Euclidean distance between adjacent base-pair centers (Angstrom); descriptive, not a standardized helical rise",
        "global_axis_projection_definition": "projection of adjacent base-pair center displacement onto the global PCA axis (Angstrom)",
        "twist_definition": "signed rotation of projected chain-A radial vector about the global axis (degrees)",
        "local_twist_definition": "signed rotation of projected chain-A radial vector about each adjacent-center step vector (degrees)",
        "global_axis_unit": axis.tolist(),
        "base_pair_centers_A": centers.tolist(),
        "center_step_distance_by_step_A": segment_lengths.tolist(),
        "center_step_distance_mean_A": float(np.mean(segment_lengths)),
        "center_step_distance_sd_A": float(np.std(segment_lengths, ddof=1)),
        "global_axis_projection_by_step_A": global_axis_projections.tolist(),
        "global_axis_projection_mean_A": float(np.mean(global_axis_projections)),
        "global_axis_projection_sd_A": float(np.std(global_axis_projections, ddof=1)),
        "centerline_bend_angle_by_step_deg": bend_angles.tolist(),
        "centerline_bend_angle_mean_deg": float(np.mean(bend_angles)),
        "centerline_bend_angle_max_deg": float(np.max(bend_angles)),
        "twist_by_step_deg": twists,
        "twist_mean_signed_deg": float(np.mean(twists)),
        "twist_mean_absolute_deg": float(np.mean(np.abs(twists))),
        "twist_sd_deg": float(np.std(twists, ddof=1)),
        "local_twist_by_step_deg": local_twists,
        "local_twist_mean_signed_deg": float(np.mean(local_twists)),
        "local_twist_sd_deg": float(np.std(local_twists, ddof=1)),
        "pair_width_by_basepair_A": pair_widths,
        "pair_width_mean_A": float(np.mean(pair_widths)),
        "base_pair_plane_to_global_axis_angle_deg": plane_global_axis_angles,
        "base_pair_plane_to_global_axis_angle_mean_deg": float(np.mean(plane_global_axis_angles)),
        "base_pair_plane_to_local_tangent_angle_deg": plane_local_tangent_angles,
        "base_pair_plane_to_local_tangent_angle_mean_deg": float(np.mean(plane_local_tangent_angles)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[2] / "artifacts/rna_pilot/reference/ion_series"
    parser.add_argument("--input-dir", type=Path, default=default_root)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.input_dir.resolve()
    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": str(root),
        "method_limits": (
            "Descriptive base-centroid/PCA metrics only; these are not standard 3DNA/CEHS parameters, "
            "not a density score, and not an MDNA model validation."
        ),
        "entries": [audit_entry(root / f"{entry_id}.cif") for entry_id in ENTRY_IDS],
    }
    output = args.output or (root / "geometry_comparison.json")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output}")
    for entry in result["entries"]:
        print(
            f"{entry['entry_id']}: center-step={entry['center_step_distance_mean_A']:.3f}±{entry['center_step_distance_sd_A']:.3f} Å, "
            f"global-axis-projection={entry['global_axis_projection_mean_A']:.3f}±{entry['global_axis_projection_sd_A']:.3f} Å, "
            f"global/local twist={entry['twist_mean_signed_deg']:.2f}/{entry['local_twist_mean_signed_deg']:.2f}°, "
            f"local-plane-angle={entry['base_pair_plane_to_local_tangent_angle_mean_deg']:.2f}°, "
            f"max centerline bend={entry['centerline_bend_angle_max_deg']:.2f}°, "
            f"pair-width={entry['pair_width_mean_A']:.2f} Å, "
            f"global-plane-angle={entry['base_pair_plane_to_global_axis_angle_mean_deg']:.2f}°"
        )


if __name__ == "__main__":
    main()
