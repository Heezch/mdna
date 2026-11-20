"""Local RNA stereochemistry and contacts against an observed model context."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


RIBOSE = {
    "C1'": ("O4'", "C2'", ("N9", "N1")),
    "C2'": ("C1'", "C3'", ("O2'",)),
    "C3'": ("C2'", "C4'", ("O3'",)),
    "C4'": ("C3'", "O4'", ("C5'",)),
}


def ribose_signs(coordinates, base, *, minimum_volume_A3=.05):
    """Use fixed substituent order and the sign of a scalar triple product."""
    result = {}
    for center, (a, b, alternatives) in RIBOSE.items():
        third = alternatives[0] if len(alternatives) == 1 or base in "AG" else alternatives[1]
        names = (center, a, b, third)
        if not all(name in coordinates for name in names):
            result[center] = None
            continue
        origin, *other = (np.asarray(coordinates[name], dtype=float) for name in names)
        volume = np.linalg.det(np.asarray(other)-origin)
        result[center] = int(np.sign(volume)) if abs(volume) >= minimum_volume_A3 else None
    return result


def contextual_contacts(atoms, environment, *, excluded=(), cutoff_A=2.):
    """Count heavy-atom pairs; only caller-identified covalent pairs are exempt."""
    if not atoms or not environment:
        return dict(contacts=[], cutoff_A=cutoff_A, excluded_pairs=len(excluded))
    xyz = np.asarray([v for _, v in environment])
    tree = cKDTree(xyz)
    excluded = set(excluded)
    contacts = []
    for key, pos in atoms:
        pos = np.asarray(pos)
        for j in sorted(tree.query_ball_point(pos, cutoff_A)):
            env_id, _ = environment[j]
            distance = float(np.linalg.norm(pos-xyz[j]))
            if distance < cutoff_A and (key, env_id) not in excluded:
                contacts.append(dict(fragment_atom=key, environment_atom=env_id, distance_A=distance))
    return dict(contacts=contacts, cutoff_A=cutoff_A, excluded_pairs=len(excluded))


def frame_residuals(observed, target):
    """Per-residue differences for frames stored as origin and row axes."""
    shift = np.linalg.norm(observed[:, 0] - target[:, 0], axis=1)
    relative = observed[:, 1:] @ target[:, 1:].transpose(0, 2, 1)
    angle = Rotation.from_matrix(relative).magnitude()
    return shift, np.degrees(angle)


def fit_fragment_boundaries(xyz, indices, fixed, target_A, *, max_shift_A=.35,
                            frames=None, target_frames=None, frame_limits=None):
    """Reduce both boundary residuals with a single bounded rigid transform."""
    xyz, fixed = np.asarray(xyz, dtype=float), np.asarray(fixed, dtype=float)
    center = xyz.mean(axis=0)

    def pose(x):
        return Rotation.from_rotvec(x[:3]).apply(xyz-center) + center + x[3:]

    def distances(pts):
        return np.linalg.norm(pts[indices]-fixed, axis=1)

    def cost(x):
        p = pose(x)
        return float(np.sum((distances(p)-target_A)**2) + .05*np.mean(np.sum((p-xyz)**2, axis=1)))

    def budgets(x):
        atom_budget = max_shift_A**2 - np.sum((pose(x)-xyz)**2, axis=1)
        if frame_limits is None:
            return atom_budget
        moved = frames.copy()
        moved[:, 0] = Rotation.from_rotvec(x[:3]).apply(frames[:, 0]-center) + center + x[3:]
        moved[:, 1:] = Rotation.from_rotvec(x[:3]).apply(
            frames[:, 1:].reshape(-1, 3)).reshape(-1, 3, 3)
        translation, rotation = frame_residuals(moved, target_frames)
        return np.r_[atom_budget, 1-(translation/frame_limits[0])**2,
                     1-(rotation/frame_limits[1])**2]

    zero = np.zeros(6)
    fit = minimize(cost, zero, method="SLSQP", constraints=[dict(type="ineq", fun=budgets)],
                   options=dict(ftol=1e-11, maxiter=100))
    trial = pose(fit.x)
    shift = float(np.linalg.norm(trial-xyz, axis=1).max())
    return trial, dict(converged=bool(fit.success), before_A=distances(xyz).tolist(),
                       after_A=distances(trial).tolist(), target_A=float(target_A),
                       max_shift_A=shift, shift_cap_A=max_shift_A)
