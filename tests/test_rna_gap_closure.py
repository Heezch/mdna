import numpy as np

from mdna import close_rna_gap_from_anchors
from mdna.rna_fragments import RNAFragmentLibrary


def _frame(x):
    return np.vstack(([x, 0.0, 0.0], np.eye(3)))


def _suite(sequence, suite_id):
    atoms = []
    base_atoms = {
        "C1'": ("C", [0.0, 0.0, 0.0]),
        "O2'": ("O", [0.0, 1.0, 0.0]),
        "N1": ("N", [0.0, 0.0, 1.0]),
        "C2": ("C", [0.0, -1.0, 1.0]),
        "P": ("P", [0.0, 0.0, -1.0]),
        "O3'": ("O", [1.4, 0.0, -1.0]),
    }
    for residue_index in (0, 1):
        offset = np.array([3.0 * residue_index, 0.0, 0.0])
        for atom_name, (element, xyz) in base_atoms.items():
            atoms.append({
                "residue_index": residue_index,
                "atom_name": atom_name,
                "element": element,
                "xyz_local_A": (np.asarray(xyz) + offset).tolist(),
            })
    return {
        "suite_id": suite_id,
        "sequence": sequence,
        "coordinate_unit": "angstrom",
        "residue_states": [
            "5prime_phosphodiester+3prime_phosphodiester",
            "5prime_phosphodiester+3prime_phosphodiester",
        ],
        "atoms": atoms,
        "bonds": [{
            "residue_index_1": 0,
            "atom_1": "O3'",
            "residue_index_2": 1,
            "atom_2": "P",
            "order": "single",
        }],
        "downstream_base_frame_relative_to_upstream": {
            "translation_A": [3.0, 0.0, 0.0],
            "rotation_matrix": np.eye(3).tolist(),
        },
    }


def _anchor_atoms(x, *, end_phosphate_y=0.0):
    return {
        "C1'": [x, 0.0, 0.0],
        "O2'": [x, 1.0, 0.0],
        "N1": [x, 0.0, 1.0],
        "C2": [x, -1.0, 1.0],
        "P": [x, end_phosphate_y, -1.0],
        "O3'": [x + 1.4, 0.0, -1.0],
    }


def test_balanced_path_shares_translation_without_stretching_suite():
    library = RNAFragmentLibrary([_suite("AC", "source:AC")])
    placement = library.place_suite(library.get_suites("AC")[0], _frame(0.),
                                    coordinate_unit="angstrom", downstream_frame=_frame(3.8),
                                    balance_frames=True)
    assert np.isclose(placement.translation_residual_A, .4)
    coords = {(a["residue_index"], a["atom_name"]): a["xyz_A"] for a in placement.atoms}
    assert np.isclose(coords[0, "C1'"][0], .4)
    assert np.isclose(coords[1, "C1'"][0]-coords[0, "C1'"][0], 3.)


def test_explicit_shared_coordinates_are_checked_against_source_bonds():
    import pytest

    lib = RNAFragmentLibrary([_suite("AC", "source:AC"), _suite("CG", "source:CG")])
    frames = np.stack([_frame(x) for x in (0., 3., 6.)])
    path = lib.select_suite_path("ACG", frames, coordinate_unit="angstrom",
                                 max_translation_residual_A=.5, max_rotation_residual_degrees=6.,
                                 max_shared_residue_rmsd_A=.5)
    coords = {a["atom_name"]: a["xyz_A"] for a in path.placements[0].atoms if a["residue_index"] == 1}
    options = dict(max_shared_atom_displacement_A=.5, max_bond_length_deviation_A=.1)
    assert lib.assemble_suite_path(path, shared_coordinates={1: coords}, **options).sequence == "ACG"
    shifted = {n: np.asarray(x)+[.2, 0, 0] for n, x in coords.items()}
    with pytest.raises(ValueError, match="assembled bond length"):
        lib.assemble_suite_path(path, shared_coordinates={1: shifted}, **options)


def test_path_diagnostics_capture_frame_rejection_without_changing_limits():
    import pytest
    from mdna.rna_fragments import NoCompatibleRNAFragmentPathError

    library = RNAFragmentLibrary([_suite("AC", "source:AC")])
    diagnostics = {}
    with pytest.raises(NoCompatibleRNAFragmentPathError):
        library.select_suite_path("AC", np.stack([_frame(0.), _frame(4.)]),
                                  coordinate_unit="angstrom", max_translation_residual_A=.5,
                                  max_rotation_residual_degrees=6., max_shared_residue_rmsd_A=.5,
                                  diagnostics=diagnostics)
    assert diagnostics["failure"] == dict(stage="frame_residual", step=0, pair="AC")
    row, = diagnostics["frame_steps"]
    assert row["translation_only_failed"] == 1
    assert row["closest"]["max_limit_ratio"] == 2.


def _close(library, *, end_x=6.0, end_atoms=None, max_search_states=100,
           max_returned_candidates=4, max_anchor_atom_displacement_A=0.01,
           environment_atoms=None, min_environment_distance_A=None,
           excluded_environment_pairs=(), pin_anchor_coordinates=False,
           deduplicate_missing_geometry=False, anchor_angle_windows_deg=None):
    return close_rna_gap_from_anchors(
        library,
        "ACG",
        _frame(0.0),
        _frame(end_x),
        _anchor_atoms(0.0),
        _anchor_atoms(end_x) if end_atoms is None else end_atoms,
        coordinate_unit="angstrom",
        max_translation_residual_A=0.05,
        max_rotation_residual_degrees=1.0,
        max_shared_residue_rmsd_A=0.01,
        max_shared_atom_displacement_A=0.01,
        max_bond_length_deviation_A=0.01,
        max_anchor_atom_displacement_A=max_anchor_atom_displacement_A,
        max_anchor_connection_length_deviation_A=0.1,
        max_search_states=max_search_states,
        max_returned_candidates=max_returned_candidates,
        environment_atoms=environment_atoms,
        min_environment_distance_A=min_environment_distance_A,
        excluded_environment_pairs=excluded_environment_pairs,
        pin_anchor_coordinates=pin_anchor_coordinates,
        deduplicate_missing_geometry=deduplicate_missing_geometry,
        anchor_angle_windows_deg=anchor_angle_windows_deg,
    )


def test_dedup_preserves_best_rank_and_reports_raw_path_count():
    library = RNAFragmentLibrary([_suite("AC", "source:z"), _suite("AC", "source:a"),
                                  _suite("CG", "source:CG")])
    result = _close(library, max_returned_candidates=1, pin_anchor_coordinates=True,
                    deduplicate_missing_geometry=True)
    assert result.total_compatible_closures == 2
    assert result.distinct_missing_geometries == 1
    assert len(result.candidates) == 1
    assert result.candidates[0].suite_ids[0] == "source:a"
    assert not result.candidates_truncated


def test_pinned_angle_screen_rejects_missing_geometry_and_requires_pinning():
    import pytest

    library = RNAFragmentLibrary([_suite("AC", "source:AC"), _suite("CG", "source:CG")])
    limits = {"o3_p_o5_deg": (90., 130.)}
    with pytest.raises(ValueError, match="requires pinned"):
        _close(library, anchor_angle_windows_deg=limits)
    result = _close(library, pin_anchor_coordinates=True, anchor_angle_windows_deg=limits)
    assert not result.candidates
    assert result.rejections["pinned_anchor_angles"] == 1
    assert result.to_report()["anchor_angle_windows_deg"] == limits


def test_pin_exact_coordinates_and_reject_excess_bond_change():
    library = RNAFragmentLibrary([_suite("AC", "source:AC"), _suite("CG", "source:CG")])
    end = _anchor_atoms(6., end_phosphate_y=.03)
    result = _close(library, end_atoms=end, max_anchor_atom_displacement_A=.1,
                    pin_anchor_coordinates=True)
    atom = next(a for a in result.candidates[0].assembly.atoms
                if a["residue_index"] == 2 and a["atom_name"] == "P")
    assert atom["xyz_A"] == end["P"]
    assert result.to_report()["anchors_pinned"] is True
    failed = _close(library, end_atoms=_anchor_atoms(6., end_phosphate_y=.25),
                    max_anchor_atom_displacement_A=.3, pin_anchor_coordinates=True)
    assert not failed.candidates
    assert failed.rejections["pinned_anchor_geometry"] == 1


def test_bounded_two_anchor_search_builds_and_reports_an_exact_short_rna_gap():
    library = RNAFragmentLibrary([
        _suite("AC", "source:AC"),
        _suite("CG", "source:CG"),
    ])

    result = _close(library)

    assert result.status == "complete"
    assert result.search_complete
    assert result.total_compatible_closures == 1
    assert result.candidates[0].suite_ids == ("source:AC", "source:CG")
    assert result.candidates[0].assembly.sequence == "ACG"
    assert result.candidates[0].assembly.shared_residue_max_displacement_A <= 0.01
    assert result.candidates[0].start_anchor_max_displacement_A <= 0.01
    assert result.candidates[0].end_anchor_max_displacement_A <= 0.01
    assert result.candidates[0].endpoint_translation_residual_A <= 0.05
    assert result.candidates[0].endpoint_rotation_residual_degrees <= 1.0
    report = result.to_report()
    assert report["status"] == "complete"
    assert len(report["start_anchor_atoms_sha256"]) == 64
    assert len(report["end_anchor_atoms_sha256"]) == 64
    components = report["candidates"][0]["score_components"]
    assert set(components) == {
        "endpoint_translation",
        "endpoint_rotation",
        "shared_residue_continuity",
        "start_anchor_deviation",
        "end_anchor_deviation",
        "start_anchor_connection_bond_deviation",
        "end_anchor_connection_bond_deviation",
        "assembled_shared_atom_deviation",
        "assembled_bond_length_deviation",
    }
    assert sum(components.values()) == report["candidates"][0]["normalized_score"]
    assert np.isclose(report["candidates"][0]["start_anchor_connection_length_A"], 1.6)
    assert np.isclose(report["candidates"][0]["end_anchor_connection_length_A"], 1.6)
    assert np.isclose(report["candidates"][0]["start_anchor_connection_reference_length_A"], 1.6)
    assert np.isclose(report["candidates"][0]["end_anchor_connection_reference_length_A"], 1.6)


def test_gap_search_distinguishes_exhaustive_no_solution_from_budget_exhaustion():
    library = RNAFragmentLibrary([
        _suite("AC", "source:AC"),
        _suite("CG", "source:CG"),
    ])

    no_solution = _close(library, end_x=7.0)
    limited = _close(library, end_x=7.0, max_search_states=1)

    assert no_solution.status == "no_solution_found"
    assert no_solution.search_complete
    assert no_solution.candidates == ()
    assert limited.status == "search_budget_exhausted"
    assert not limited.search_complete
    assert limited.candidates == ()


def test_gap_search_keeps_multiple_closures_in_stable_score_order():
    library = RNAFragmentLibrary([
        _suite("AC", "source:AC:a"),
        _suite("AC", "source:AC:b"),
        _suite("CG", "source:CG:a"),
        _suite("CG", "source:CG:b"),
    ])

    result = _close(library, max_returned_candidates=2)

    assert result.search_complete
    assert result.total_compatible_closures == 4
    assert result.candidates_truncated
    assert [candidate.suite_ids for candidate in result.candidates] == [
        ("source:AC:a", "source:CG:a"),
        ("source:AC:a", "source:CG:b"),
    ]


def test_gap_search_rejects_candidate_that_moves_a_fixed_anchor_atom():
    library = RNAFragmentLibrary([
        _suite("AC", "source:AC"),
        _suite("CG", "source:CG"),
    ])
    end_atoms = _anchor_atoms(6.0)
    end_atoms["O2'"] = [6.0, 1.2, 0.0]

    result = _close(library, end_atoms=end_atoms)

    assert result.status == "no_solution_found"
    assert result.search_complete
    assert result.rejections["anchor_displacement"] == 1


def test_gap_search_checks_connections_after_restoring_fixed_anchor_coordinates():
    library = RNAFragmentLibrary([
        _suite("AC", "source:AC"),
        _suite("CG", "source:CG"),
    ])
    end_atoms = _anchor_atoms(6.0, end_phosphate_y=1.0)

    result = _close(
        library,
        end_atoms=end_atoms,
        max_anchor_atom_displacement_A=2.0,
    )

    assert result.status == "no_solution_found"
    assert result.search_complete
    assert result.rejections["anchor_connection_geometry"] == 1


def test_gap_search_rejects_missing_residue_atoms_too_close_to_environment():
    library = RNAFragmentLibrary([
        _suite("AC", "source:AC"),
        _suite("CG", "source:CG"),
    ])

    result = _close(
        library,
        environment_atoms=[{"atom_id": "protein:42:CA", "element": "C", "xyz_A": [3.0, 0.0, 0.25]}],
        min_environment_distance_A=0.5,
    )

    assert result.status == "no_solution_found"
    assert result.search_complete
    assert result.rejections["environment_clash"] == 1


def test_gap_search_reports_environment_clearance_and_honors_explicit_covalent_exclusion():
    library = RNAFragmentLibrary([
        _suite("AC", "source:AC"),
        _suite("CG", "source:CG"),
    ])

    result = _close(
        library,
        environment_atoms=[{"atom_id": "anchor:O3'", "element": "O", "xyz_A": [3.0, 0.0, 0.0]}],
        min_environment_distance_A=0.5,
        excluded_environment_pairs=((1, "C1'", "anchor:O3'"),),
    )

    assert result.status == "complete"
    candidate = result.candidates[0]
    assert np.isclose(candidate.environment_min_distance_A, 1.0)
    report = result.to_report()
    assert len(report["environment_atoms_sha256"]) == 64
    assert np.isclose(report["candidates"][0]["environment_min_distance_A"], 1.0)
    assert report["rejections"]["environment_clash"] == 0


def test_gap_search_requires_an_explicit_positive_cutoff_when_environment_is_supplied():
    import pytest

    library = RNAFragmentLibrary([_suite("AC", "source:AC"), _suite("CG", "source:CG")])

    with pytest.raises(ValueError, match="min_environment_distance_A"):
        _close(library, environment_atoms=[{
            "atom_id": "x", "element": "C", "xyz_A": [0.0, 0.0, 0.0]
        }])


def test_gap_search_rejects_hydrogen_in_heavy_atom_environment():
    import pytest

    library = RNAFragmentLibrary([_suite("AC", "source:AC"), _suite("CG", "source:CG")])

    with pytest.raises(ValueError, match="non-hydrogen element"):
        _close(
            library,
            environment_atoms=[{"atom_id": "water:H1", "element": "H", "xyz_A": [0, 0, 0]}],
            min_environment_distance_A=1.5,
        )


def test_gap_search_rejects_covalent_exclusion_for_atom_absent_from_gap():
    import pytest

    library = RNAFragmentLibrary([_suite("AC", "source:AC"), _suite("CG", "source:CG")])

    with pytest.raises(ValueError, match="not present in the gap atom inventory"):
        _close(
            library,
            environment_atoms=[{"atom_id": "anchor:O3'", "element": "O", "xyz_A": [3, 0, 0]}],
            min_environment_distance_A=1.5,
            excluded_environment_pairs=((1, "DOES_NOT_EXIST", "anchor:O3'"),),
        )
