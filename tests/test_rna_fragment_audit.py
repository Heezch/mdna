import numpy as np
import pytest

from scripts.rna_pilot.evaluate_ccd_to_polymer_fit import rigid_fit
from scripts.rna_pilot.extract_rna_suite_library import (
    make_base_frame,
    summarize_conformer_spread,
    validate_suite,
)


def test_rigid_fit_recovers_a_proper_rotation_and_translation():
    source = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.5, 0.5, 1.0],
        ]
    )
    rotation_expected = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation_expected = np.array([4.0, -2.0, 7.0])
    target = source @ rotation_expected + translation_expected

    rotation, translation = rigid_fit(source, target)

    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert rotation == pytest.approx(rotation_expected)
    assert translation == pytest.approx(translation_expected)
    assert source @ rotation + translation == pytest.approx(target)


def test_rigid_fit_rejects_nonfinite_coordinates():
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    target = source.copy()
    target[0, 1] = np.nan

    with pytest.raises(ValueError, match="coordinates must be finite"):
        rigid_fit(source, target)


def test_rigid_fit_rejects_collinear_points_with_ambiguous_orientation():
    source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="do not define a unique rigid orientation"):
        rigid_fit(source, source)


def _synthetic_cytosine_coordinates():
    return {
        "N1": np.array([1.0, 0.0, 0.0]),
        "C2": np.array([0.5, 0.866, 0.0]),
        "N3": np.array([-0.5, 0.866, 0.0]),
        "C4": np.array([-1.0, 0.0, 0.0]),
        "C5": np.array([-0.5, -0.866, 0.0]),
        "C6": np.array([0.5, -0.866, 0.0]),
        "C1'": np.array([1.4, 0.0, 1.2]),
    }


def test_rna_base_frame_is_right_handed_and_rigid_transform_invariant():
    coordinates = _synthetic_cytosine_coordinates()
    origin, frame = make_base_frame("C", coordinates)
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([3.0, -4.0, 2.0])
    transformed = {name: xyz @ rotation + translation for name, xyz in coordinates.items()}

    transformed_origin, transformed_frame = make_base_frame("C", transformed)

    assert np.linalg.det(frame) == pytest.approx(1.0)
    assert frame.T @ frame == pytest.approx(np.eye(3))
    assert transformed_origin == pytest.approx(origin @ rotation + translation)
    assert transformed_frame == pytest.approx(rotation.T @ frame)


def test_rna_base_frame_rejects_unknown_base_and_missing_anchors():
    with pytest.raises(ValueError, match="Unsupported RNA base"):
        make_base_frame("X", {})
    with pytest.raises(ValueError, match="lacks base-frame atoms"):
        make_base_frame("C", {"N1": np.array([0.0, 0.0, 0.0])})


def _minimal_suite():
    return {
        "suite_id": "unit:A:1-2",
        "sequence": "AC",
        "residue_states": [
            "5prime_phosphodiester+3prime_phosphodiester",
            "5prime_phosphodiester+3prime_phosphodiester",
        ],
        "atoms": [
            {"residue_index": 0, "atom_name": "O3'", "xyz_local_A": [0.0, 0.0, 0.0]},
            {"residue_index": 1, "atom_name": "P", "xyz_local_A": [1.6, 0.0, 0.0]},
        ],
        "bonds": [
            {
                "residue_index_1": 0,
                "atom_1": "O3'",
                "residue_index_2": 1,
                "atom_2": "P",
                "order": "single",
            }
        ],
        "observed_O3_prime_to_next_P_A": 1.6,
        "downstream_base_frame_relative_to_upstream": {
            "rotation_matrix": np.eye(3).tolist()
        },
    }


def test_suite_validator_accepts_complete_fragment_and_rejects_missing_bond_atom():
    suite = _minimal_suite()
    validate_suite(suite)
    suite["bonds"][0]["atom_2"] = "missing"

    with pytest.raises(ValueError, match="bond endpoint is absent"):
        validate_suite(suite)


def test_conformer_spread_summary_compares_same_sequence_coordinates():
    first = _minimal_suite()
    second = _minimal_suite()
    second["suite_id"] = "unit:B:1-2"
    second["atoms"][1]["xyz_local_A"] = [2.0, 0.0, 0.0]

    result = summarize_conformer_spread([first, second])

    assert result["AC"]["suite_count"] == 2
    assert result["AC"]["pairwise_conformer_rmsd_count"] == 1
    assert result["AC"]["pairwise_conformer_rmsd_A"]["median"] == pytest.approx(
        np.sqrt(0.16 / 2)
    )
