import numpy as np

import mdna


def test_frames_base_pairs_and_splines_are_public():
    for name in ("Nucleic", "SplineFrames", "BP_MAP"):
        assert name in mdna.__all__ and hasattr(mdna, name)
    assert all(mdna.BP_MAP[mdna.BP_MAP[b]] == b for b in "ATGCBSZPLM")


def test_frames_layout_matches_its_documentation():
    dna = mdna.make(sequence="GACGTTCA")
    frames = np.asarray(dna.get_frames())
    assert frames.shape == (8, 1, 4, 3)
    step = np.diff(frames[:, -1, 0], axis=0)
    assert np.allclose(np.linalg.norm(step, axis=1), 0.34, atol=0.02)  # origins in nm
    assert (np.einsum("ij,ij->i", step / np.linalg.norm(step, axis=1)[:, None], frames[:-1, -1, 3]) > 0.98).all()  # z is the axis


def test_both_relaxation_modes_record_energy_per_cycle():
    ring = np.c_[3 * np.cos(np.linspace(0, 2 * np.pi, 8, endpoint=False)), 3 * np.sin(np.linspace(0, 2 * np.pi, 8, endpoint=False)), np.zeros(8)]
    for simple in (False, True):
        dna = mdna.make(control_points=ring, circular=True, n_bp=60)
        dna.minimize(simple=simple, equilibrate_writhe=False)
        energies = np.asarray(dna.minimizer.out["energies"])
        assert energies.ndim == 2 and len(energies) > 1 and (np.diff(energies[:, 0]) > 0).all(), (simple, energies.shape)
