"""Tests for the writhe-corrected ΔLk feature.

The White–Fuller theorem states Lk = Tw + Wr.  When the user requests a
linking number difference ``dLk`` via ``mdna.make(..., dLk=X)``, the twist
must account for the pre-existing writhe of the backbone shape so that the
*total* linking number satisfies  Lk = Tw₀ + dLk  (where Tw₀ is the
intrinsic/relaxed twist).
"""

import numpy as np
import pytest
import tempfile
import os

import mdna


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _toroidal_helix(R: float = 8.0, r: float = 3.0, n_winds: int = 2,
                    n_pts: int = 80) -> np.ndarray:
    """Return control points for a toroidal helix with significant writhe."""
    t = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    return np.column_stack([
        (R + r * np.cos(n_winds * t)) * np.cos(t),
        (R + r * np.cos(n_winds * t)) * np.sin(t),
        r * np.sin(n_winds * t),
    ])


# ---------------------------------------------------------------------------
# Linking number consistency tests
# ---------------------------------------------------------------------------

class TestWritheCorrectedDLk:
    """Verify that dLk correctly accounts for pre-existing writhe."""

    def test_planar_circle_dLk_zero(self):
        """For a planar circle Wr≈0, so dLk=0 should give Lk ≈ Tw₀."""
        shape = mdna.Shapes.circle(radius=2)
        dna = mdna.make(sequence="A" * 100, control_points=shape,
                        circular=True, dLk=0)
        lk, wr, tw = dna.get_linking_number()

        assert abs(wr) < 0.05, f"Planar circle writhe should be ~0, got {wr:.4f}"
        assert abs(lk - tw) < 0.05, f"Lk ({lk:.2f}) should equal Tw ({tw:.2f}) for Wr≈0"

    def test_planar_circle_dLk_positive(self):
        """dLk=2 on a planar circle should increase Lk by 2."""
        shape = mdna.Shapes.circle(radius=2)
        dna0 = mdna.make(sequence="A" * 100, control_points=shape,
                         circular=True, dLk=0)
        dna2 = mdna.make(sequence="A" * 100, control_points=shape,
                         circular=True, dLk=2)

        lk0 = dna0.get_linking_number()[0]
        lk2 = dna2.get_linking_number()[0]

        assert abs((lk2 - lk0) - 2) < 0.1, (
            f"ΔLk should be 2, got {lk2 - lk0:.2f}"
        )

    def test_toroidal_helix_dLk_zero_white_fuller(self):
        """For a non-planar curve with Wr≠0, Lk = Tw + Wr must hold."""
        pts = _toroidal_helix()
        dna = mdna.make(sequence="A" * 300, control_points=pts,
                        circular=True, dLk=0)
        lk, wr, tw = dna.get_linking_number()

        assert abs(lk - (tw + wr)) < 0.1, (
            f"White–Fuller violated: Lk={lk:.2f}, Tw+Wr={tw + wr:.2f}"
        )

    def test_toroidal_helix_dLk_shifts_linking_number(self):
        """Increasing dLk by 2 should increase Lk by 2 regardless of writhe."""
        pts = _toroidal_helix()
        dna0 = mdna.make(sequence="A" * 300, control_points=pts,
                         circular=True, dLk=0)
        dna2 = mdna.make(sequence="A" * 300, control_points=pts,
                         circular=True, dLk=2)

        lk0 = dna0.get_linking_number()[0]
        lk2 = dna2.get_linking_number()[0]

        assert abs((lk2 - lk0) - 2) < 0.2, (
            f"ΔLk should be 2, got {lk2 - lk0:.2f}"
        )

    def test_toroidal_helix_writhe_is_nonzero(self):
        """Ensure the toroidal helix fixture actually has meaningful writhe."""
        pts = _toroidal_helix()
        dna = mdna.make(sequence="A" * 300, control_points=pts,
                        circular=True, dLk=0)
        _, wr, _ = dna.get_linking_number()

        assert abs(wr) > 0.1, (
            f"Toroidal helix should have |Wr| > 0.1, got {wr:.4f}"
        )


# ---------------------------------------------------------------------------
# Linear DNA
# ---------------------------------------------------------------------------

class TestLinearDLk:
    """dLk on linear (non-circular) DNA."""

    def test_linear_dLk_does_not_crash(self):
        """Linear DNA with dLk should run without error."""
        dna = mdna.make(sequence="A" * 50, dLk=1)
        assert dna.n_bp == 50

    def test_linear_no_dLk_default(self):
        """Without dLk, default twist should be ~34.3°/bp."""
        dna = mdna.make(sequence="A" * 20)
        assert dna.n_bp == 20


# ---------------------------------------------------------------------------
# Save / Load round-trip preserves Lk
# ---------------------------------------------------------------------------

class TestSaveLoadRoundTrip:
    """Verify that saving to PDB and reloading preserves linking number."""

    def test_circular_dLk_roundtrip(self):
        """Lk should be preserved through PDB save → load."""
        shape = mdna.Shapes.circle(radius=2)
        dna = mdna.make(sequence="A" * 100, control_points=shape,
                        circular=True, dLk=2)
        lk_before = dna.get_linking_number()

        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_stem = os.path.join(tmpdir, "test_circular")
            dna.save_pdb(filename=pdb_stem)  # writes pdb_stem + ".pdb"

            dna_loaded = mdna.load(filename=pdb_stem + ".pdb", circular=True)
            lk_after = dna_loaded.get_linking_number()

        # Lk should be conserved to within ~0.5 (small numerical drift from
        # PDB coordinate precision — 3 decimal places ≈ 0.001 nm)
        assert abs(lk_before[0] - lk_after[0]) < 0.5, (
            f"Lk changed through save/load: {lk_before[0]:.2f} → {lk_after[0]:.2f}"
        )

    def test_toroidal_save_load_preserves_writhe(self):
        """Writhe should survive PDB round-trip within tolerance."""
        pts = _toroidal_helix()
        dna = mdna.make(sequence="A" * 300, control_points=pts,
                        circular=True, dLk=0)
        lk_before = dna.get_linking_number()

        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_stem = os.path.join(tmpdir, "test_toroidal")
            dna.save_pdb(filename=pdb_stem)  # writes pdb_stem + ".pdb"

            dna_loaded = mdna.load(filename=pdb_stem + ".pdb", circular=True)
            lk_after = dna_loaded.get_linking_number()

        assert abs(lk_before[1] - lk_after[1]) < 0.5, (
            f"Writhe changed through save/load: {lk_before[1]:.2f} → {lk_after[1]:.2f}"
        )
