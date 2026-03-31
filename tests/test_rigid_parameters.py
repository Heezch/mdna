"""Tests for rigid base-pair and step parameter computation against 3DNA reference.

Validates that NucleicFrames produces results consistent with the 3DNA
analysis of the nucleosome crystal structure 1KX5 (147 bp).
"""

import os
import numpy as np
import pytest
import mdtraj as md
import mdna

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(__file__)
_PDB_PATH = os.path.join(_HERE, '..', 'docs', 'notebooks', 'pdbs', '1kx5.pdb')
_3DNA_PATH = os.path.join(_HERE, 'data', '1kx5-3dna.out')

# ---------------------------------------------------------------------------
# Helpers – 3DNA output parsers
# ---------------------------------------------------------------------------

def _normalize_parameter_name(parameter_name):
    key = ''.join(ch for ch in parameter_name.lower() if ch.isalnum())
    alias_map = {
        'shift': {'shift', 'dx'}, 'slide': {'slide', 'dy'}, 'rise': {'rise', 'dz'},
        'tilt': {'tilt', 'rx'}, 'roll': {'roll', 'ry'}, 'twist': {'twist', 'rz'},
        'shear': {'shear'}, 'stretch': {'stretch'}, 'stagger': {'stagger'},
        'buckle': {'buckle'}, 'propeller': {'propeller', 'propellor', 'prop'},
        'opening': {'opening'},
    }
    for canonical, aliases in alias_map.items():
        if key in aliases:
            return canonical
    return key


def _parse_3dna_section(file_path, section_marker):
    """Generic parser for 3DNA tabular sections."""
    in_section = False
    columns = None
    rows = []

    with open(file_path, 'r') as handle:
        for line in handle:
            stripped = line.strip()

            if section_marker in line:
                in_section = True
                continue
            if not in_section:
                continue
            if stripped.startswith('****'):
                if rows:
                    break
                continue
            if stripped.startswith('bp'):
                header_left = line.split('#', 1)[0]
                header_tokens = header_left.split()
                columns = [_normalize_parameter_name(t) for t in header_tokens[1:]]
                continue
            if columns is None or not stripped or stripped.startswith('#'):
                continue
            data_left = line.split('#', 1)[0]
            tokens = data_left.split()
            if len(tokens) < 2 + len(columns):
                continue
            numeric_tokens = tokens[2:2 + len(columns)]
            try:
                rows.append([float(t) for t in numeric_tokens])
            except ValueError:
                continue

    if not rows or columns is None:
        raise ValueError(f'Could not parse section "{section_marker}" from {file_path}')

    arr = np.asarray(rows, dtype=float)
    return {col: arr[:, idx] for idx, col in enumerate(columns)}


def _load_3dna_step_parameters(file_path):
    data = _parse_3dna_section(file_path, 'Local base-pair step parameters')
    # Prepend a zero for bp index 0 (no step before first bp)
    return {k: np.concatenate([[0.0], v]) for k, v in data.items()}


def _load_3dna_bp_parameters(file_path):
    return _parse_3dna_section(file_path, 'Simple base-pair parameters')


def _to_1d(values):
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr[0] if arr.shape[0] == 1 else arr.mean(axis=0)
    return arr.reshape(-1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def traj():
    return md.load(_PDB_PATH)


@pytest.fixture(scope='module')
def rigid(traj):
    return mdna.compute_rigid_parameters(traj, chainids=[0, 1], fit_reference=False)


@pytest.fixture(scope='module')
def ref_step():
    return _load_3dna_step_parameters(_3DNA_PATH)


@pytest.fixture(scope='module')
def ref_bp():
    return _load_3dna_bp_parameters(_3DNA_PATH)


# ---------------------------------------------------------------------------
# Step parameter tests  (shift, slide, rise, tilt, roll, twist)
# ---------------------------------------------------------------------------

# Tolerances: translational in Ångström, rotational in degrees.
# mdna uses Cayley vectors, 3DNA uses Euler angles, so we allow larger
# tolerances for the rotational parameters.
STEP_TOLERANCES = {
    'shift': {'max': 0.5, 'mean': 0.15},
    'slide': {'max': 0.3, 'mean': 0.10},
    'rise':  {'max': 0.4, 'mean': 0.10},
    'tilt':  {'max': 5.0, 'mean': 2.0},
    'roll':  {'max': 9.0, 'mean': 2.5},
    'twist': {'max': 4.0, 'mean': 1.5},
}


@pytest.mark.parametrize('name', ['shift', 'slide', 'rise', 'tilt', 'roll', 'twist'])
def test_step_parameter_vs_3dna(rigid, ref_step, name):
    """Inter base-pair step parameters should agree with 3DNA within tolerance."""
    mdna_vals = _to_1d(rigid.get_parameter(name))
    ref_vals = ref_step[name]
    n = min(len(mdna_vals), len(ref_vals))
    # Skip index 0 (zero-padded placeholder)
    diff = np.abs(mdna_vals[1:n] - ref_vals[1:n])

    tol = STEP_TOLERANCES[name]
    assert diff.max() < tol['max'], (
        f"{name}: max diff {diff.max():.4f} exceeds tolerance {tol['max']}"
    )
    assert diff.mean() < tol['mean'], (
        f"{name}: mean diff {diff.mean():.4f} exceeds tolerance {tol['mean']}"
    )


# ---------------------------------------------------------------------------
# Base-pair parameter tests  (shear, stretch, stagger, buckle, propeller, opening)
# ---------------------------------------------------------------------------

BP_TOLERANCES = {
    'shear':     {'max': 0.5, 'mean': 0.15},
    'stretch':   {'max': 0.5, 'mean': 0.15},
    'stagger':   {'max': 1.0, 'mean': 0.25},
    'buckle':    {'max': 5.0, 'mean': 2.0},
    'propeller': {'max': 15.0, 'mean': 3.0},
    'opening':   {'max': 8.0, 'mean': 2.0},
}


@pytest.mark.parametrize('name', ['shear', 'stretch', 'stagger', 'buckle', 'propeller', 'opening'])
def test_bp_parameter_vs_3dna(rigid, ref_bp, name):
    """Intra base-pair parameters should agree with 3DNA within tolerance."""
    mdna_vals = _to_1d(rigid.get_parameter(name))
    ref_vals = ref_bp[name]
    n = min(len(mdna_vals), len(ref_vals))
    diff = np.abs(mdna_vals[:n] - ref_vals[:n])

    tol = BP_TOLERANCES[name]
    assert diff.max() < tol['max'], (
        f"{name}: max diff {diff.max():.4f} exceeds tolerance {tol['max']}"
    )
    assert diff.mean() < tol['mean'], (
        f"{name}: mean diff {diff.mean():.4f} exceeds tolerance {tol['mean']}"
    )


# ---------------------------------------------------------------------------
# Frame consistency tests
# ---------------------------------------------------------------------------

def test_base_frame_return_order():
    """calculate_base_frame() must return [b_R, b_D, b_L, b_N]."""
    from mdna.geometry import ReferenceBase
    traj = md.load(_PDB_PATH)
    res = list(traj.top.chain(0).residues)[0]
    res_traj = traj.atom_slice([at.index for at in res.atoms])
    ref = ReferenceBase(res_traj)

    result = ref.calculate_base_frame()
    # The return is [b_R, b_D, b_L, b_N]
    assert result.shape[0] == 4
    # After unpacking, attributes must match
    np.testing.assert_array_equal(result[0], ref.b_R)
    np.testing.assert_array_equal(result[1], ref.b_D)
    np.testing.assert_array_equal(result[2], ref.b_L)
    np.testing.assert_array_equal(result[3], ref.b_N)


def test_base_frame_orthonormality():
    """The three frame vectors (b_D, b_L, b_N) must be orthonormal."""
    from mdna.geometry import ReferenceBase
    traj = md.load(_PDB_PATH)
    res = list(traj.top.chain(0).residues)[0]
    res_traj = traj.atom_slice([at.index for at in res.atoms])
    ref = ReferenceBase(res_traj)

    D, L, N = ref.b_D[0], ref.b_L[0], ref.b_N[0]
    # Unit vectors
    np.testing.assert_allclose(np.linalg.norm(D), 1.0, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(L), 1.0, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(N), 1.0, atol=1e-10)
    # Orthogonality (float32 precision from mdtraj coordinates)
    np.testing.assert_allclose(np.dot(D, L), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.dot(D, N), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.dot(L, N), 0.0, atol=1e-6)
    # Right-handed
    np.testing.assert_allclose(np.cross(D, L), N, atol=1e-6)


def test_get_base_vectors_ordering(traj):
    """get_base_vectors must return [b_R, b_D, b_L, b_N] ordering."""
    from mdna.geometry import NucleicFrames, ReferenceBase
    nf = NucleicFrames(traj, chainids=[0, 1], fit_reference=False)
    res = nf.res_A[0]
    res_traj = traj.atom_slice([at.index for at in res.atoms])
    ref = ReferenceBase(res_traj)
    vectors = nf.get_base_vectors(res_traj)
    # vectors shape: (n_frames, 4, 3)  →  [b_R, b_D, b_L, b_N]
    np.testing.assert_array_equal(vectors[0, 0], ref.b_R[0])
    np.testing.assert_array_equal(vectors[0, 1], ref.b_D[0])
    np.testing.assert_array_equal(vectors[0, 2], ref.b_L[0])
    np.testing.assert_array_equal(vectors[0, 3], ref.b_N[0])


def test_parameter_names(rigid):
    """Parameter names must follow the Cambridge/Tsukuba convention."""
    expected_bp = ['shear', 'stretch', 'stagger', 'buckle', 'propeller', 'opening']
    expected_step = ['shift', 'slide', 'rise', 'tilt', 'roll', 'twist']
    assert rigid.base_parameter_names == expected_bp
    assert rigid.step_parameter_names == expected_step
    assert rigid.names == expected_bp + expected_step


def test_parameter_array_shapes(rigid):
    """Output arrays must have consistent shapes."""
    n_bp = len(rigid.res_A)
    n_frames = rigid.traj.n_frames
    # Shape convention: (n_bp, n_frames, n_params)
    # For a single-frame PDB n_frames=1, axes may appear swapped.
    bp_shape = rigid.bp_params.shape
    assert bp_shape[-1] == 6
    assert n_bp in bp_shape and n_frames in bp_shape
    step_shape = rigid.step_params.shape
    assert step_shape[-1] == 6
    assert n_bp in step_shape and n_frames in step_shape
    param_shape = rigid.parameters.shape
    assert param_shape[-1] == 12
    assert n_bp in param_shape and n_frames in param_shape
    frame_shape = rigid.frames.shape
    assert frame_shape[-2:] == (4, 3)
    assert n_bp in frame_shape and n_frames in frame_shape
