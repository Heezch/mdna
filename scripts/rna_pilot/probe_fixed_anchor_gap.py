"""Adjust one missing RNA residue while keeping both observed anchors fixed."""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


FIELDS = ("o3p_A", "c3_o3_p_deg", "o3_p_o5_deg", "p_o5_c5_deg")
SCALES = np.array([.15, 15., 15., 15.])
LOW = np.array([1.3, 100., 90., 95.])
HIGH = np.array([2., 140., 130., 135.])
TORSIONS = ("c4_c3_o3_p_deg", "c3_o3_p_o5_deg", "o3_p_o5_c5_deg", "p_o5_c5_c4_deg")


def choose_trial(rigid, phosphate):
    """Use geometry-only pass flags; masked-coordinate RMSD is never a selector."""
    return ("rigid" if rigid["passed_diagnostic"] else
            "phosphate" if phosphate["passed_diagnostic"] else None)


def torsion_diagnostic(qc, reference):
    """Report nearest joint circular RMS deviation per link; descriptive only."""
    targets = np.array([[row[k] for k in TORSIONS] for row in reference])
    result = []
    for row in qc["links"]:
        if any(row[k] is None for k in TORSIONS):
            result.append(None)
            continue
        delta = (np.array([row[k] for k in TORSIONS]) - targets + 180.) % 360. - 180.
        result.append(float(np.sqrt(np.mean(delta**2, axis=1)).min()))
    return result


def link_matrix(measure, start, middle, end):
    links = (measure(start, middle), measure(middle, end))
    return np.array([[row[key] for key in FIELDS] for row in links], dtype=float)


def fit_missing_residue(measure, start, middle, end, target, *, max_shift_A=.5,
                        max_phosphate_turn_deg=0.):
    """Return a trial and measured metrics; never use masked truth in the fit."""
    names = tuple(sorted(middle))
    xyz = np.array([middle[name] for name in names], dtype=float)
    center = xyz.mean(axis=0)
    if not np.isfinite(max_phosphate_turn_deg) or not 0 <= max_phosphate_turn_deg <= 180:
        raise ValueError("expected a phosphate rotation cap in [0, 180] degrees")
    flexible = max_phosphate_turn_deg > 0
    if flexible:
        ids = [names.index(name) for name in ("P", "OP1", "OP2")]
        pivot = xyz[names.index("O5'")]
        axis = xyz[names.index("C5'")] - pivot
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            raise ValueError("cannot rotate phosphate about a degenerate O5'–C5' axis")
        axis /= norm

    def pose(x):
        pts = xyz.copy()
        if flexible:
            pts[ids] = pivot + Rotation.from_rotvec(x[6]*axis).apply(pts[ids]-pivot)
        pts = Rotation.from_rotvec(x[:3]).apply(pts-center) + center + x[3:6]
        return dict(zip(names, pts, strict=True)), pts

    def residual(x):
        moving, pts = pose(x)
        links = link_matrix(measure, start, moving, end)
        shift = np.linalg.norm(pts-xyz, axis=1)
        return np.r_[((links-target)/SCALES).ravel(),
                     .5*(pts-xyz).ravel(), 40.*np.maximum(0., shift-max_shift_A)]

    cap = np.radians(max_phosphate_turn_deg)
    if flexible:
        low, high = np.r_[np.full(6, -np.inf), -cap], np.r_[np.full(6, np.inf), cap]
        result = least_squares(residual, np.zeros(7), bounds=(low, high), max_nfev=200)
    else:
        result = least_squares(residual, np.zeros(6), max_nfev=200)
    trial, pts = pose(result.x)
    before = link_matrix(measure, start, middle, end)
    after = link_matrix(measure, start, trial, end)
    return trial, dict(converged=bool(result.success),
                       before=before.tolist(), after=after.tolist(),
                       initial_error=float(np.linalg.norm((before-target)/SCALES)),
                       final_error=float(np.linalg.norm((after-target)/SCALES)),
                       max_shift_A=float(np.linalg.norm(pts-xyz, axis=1).max()),
                       phosphate_turn_deg=float(np.degrees(result.x[6])) if flexible else 0.,
                       before_in_window=bool(np.all((before >= LOW) & (before <= HIGH))),
                       after_in_window=bool(np.all((after >= LOW) & (after <= HIGH))))
