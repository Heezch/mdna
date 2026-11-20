import copy
from pathlib import Path

import numpy as np

from mdna import DNAFragmentLibrary, close_dna_gap_from_anchors


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIBRARY_PATH = PROJECT_ROOT / "artifacts/gap_benchmarks/1kx5_dna_suites.json"


def _case():
    library = DNAFragmentLibrary.from_json(LIBRARY_PATH)
    suites = [suite for context in library.available_sequences
              for suite in library.get_suites(context)]
    first, second = next(
        (left, right) for left in suites for right in suites
        if left["chain_id"] == right["chain_id"]
        and left["source_residues"][1] == right["source_residues"][0]
    )
    sequence = first["sequence"] + second["sequence"][1]
    identity = np.eye(3)
    origins = [np.zeros(3)]
    bases = [identity]
    for suite in (first, second):
        transform = suite["downstream_base_frame_relative_to_upstream"]
        origins.append(origins[-1] + np.asarray(transform["translation_A"]) @ bases[-1])
        bases.append(np.asarray(transform["rotation_matrix"]).T @ bases[-1])
    frames = [np.vstack((origin, basis)) for origin, basis in zip(origins, bases)]
    left = library.place_suite(first, frames[0], coordinate_unit="angstrom")
    right = library.place_suite(second, frames[1], coordinate_unit="angstrom",
                                downstream_frame=frames[2])

    def anchor(placement, local_residue):
        return {
            atom["atom_name"]: atom["xyz_A"]
            for atom in placement.atoms if atom["residue_index"] == local_residue
        }

    return library, sequence, frames[0], frames[2], anchor(left, 0), anchor(right, 1), first, second


def _close(case, *, max_search_states=10000, end_frame=None, end_atoms=None,
           max_shared_atom_displacement_A=0.03):
    library, sequence, start, end, start_atoms, target_end_atoms, _, _ = case
    return close_dna_gap_from_anchors(
        library, sequence, start, end if end_frame is None else end_frame,
        start_atoms, target_end_atoms if end_atoms is None else end_atoms,
        coordinate_unit="angstrom",
        max_translation_residual_A=0.05,
        max_rotation_residual_degrees=0.5,
        max_shared_residue_rmsd_A=0.03,
        max_shared_atom_displacement_A=max_shared_atom_displacement_A,
        max_bond_length_deviation_A=0.03,
        max_anchor_atom_displacement_A=0.03,
        max_anchor_connection_length_deviation_A=0.03,
        max_search_states=max_search_states,
        max_returned_candidates=4,
    )


def test_dna_gap_search_closes_a_single_residue_between_fixed_anchors():
    case = _case()
    result = _close(case)

    assert result.status == "complete"
    assert result.search_complete
    candidate = next(item for item in result.candidates
                     if item.suite_ids == (case[6]["suite_id"], case[7]["suite_id"]))
    assert candidate.assembly.sequence == case[1]
    assert candidate.start_anchor_max_displacement_A <= 1e-8
    assert candidate.end_anchor_max_displacement_A <= 1e-8
    assert candidate.endpoint_translation_residual_A <= 0.05
    assert candidate.endpoint_rotation_residual_degrees <= 0.5
    assert candidate.assembly.five_prime_port == (0, "P")
    assert candidate.assembly.three_prime_port == (2, "O3'")

    report = result.to_report()
    assert report["scientific_acceptance"] is False
    assert len(report["start_anchor_atoms_sha256"]) == 64
    assert report["candidates"][0]["environment_screening"] == "not_evaluated"


def test_dna_gap_search_distinguishes_no_solution_from_budget_exhaustion():
    case = _case()
    unreachable = np.array(case[3], copy=True)
    unreachable[0] += np.array([10.0, 0.0, 0.0])

    no_solution = _close(case, end_frame=unreachable)
    limited = _close(case, end_frame=unreachable, max_search_states=1)

    assert no_solution.status == "no_solution_found"
    assert no_solution.search_complete
    assert not no_solution.candidates
    minima = no_solution.to_report()["trial_metric_minima"]
    assert minima["start_anchor_max_displacement_A"] <= 1e-8
    assert minima["endpoint_translation_residual_A"] >= 9.0
    assert "shared_residue_rmsd_A" not in minima
    assert limited.status == "search_budget_exhausted"
    assert not limited.search_complete


def test_dna_gap_search_rejects_a_displaced_fixed_anchor_atom():
    case = _case()
    end_atoms = copy.deepcopy(case[5])
    end_atoms["C1'"][0] += 0.5

    result = _close(case, end_atoms=end_atoms)

    assert result.status == "no_solution_found"
    assert result.search_complete
    assert result.rejections["anchor_displacement"] > 0


def test_dna_gap_report_preserves_specific_assembly_rejection_reasons():
    result = _close(_case(), max_shared_atom_displacement_A=0.0)

    assert result.rejections["assembly"] > 0
    report = result.to_report()
    assert report["assembly_rejection_reasons"]
    assert sum(report["assembly_rejection_reasons"].values()) == result.rejections["assembly"]
