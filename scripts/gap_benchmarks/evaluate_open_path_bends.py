"""Source-tree benchmark. Run with --curve-only when MDTraj is unavailable."""
import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import runpy
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261119
BUDGET_NM = 0.90
SEQUENCE = 'ACGT' * 12
# Deliberately broad triage window, used only for this benchmark. In-range
# links still require validation against chemistry/conformation-specific ideals.
O3P_SCREEN_A = (1.30, 2.00)
LINK_ANGLE_WINDOWS_DEG = {"c3_o3_p_deg": (100, 140), "o3_p_o5_deg": (90, 130),
                          "p_o5_c5_deg": (95, 135)}
LINK_TORSION_FIELDS = ("c4_c3_o3_p_deg", "c3_o3_p_o5_deg", "o3_p_o5_c5_deg", "p_o5_c5_c4_deg")


def screen_atomic_geometry(metrics):
    low, high = O3P_SCREEN_A
    chains = {}
    for row in metrics['links']:
        chains.setdefault(row['chain_index'], []).append(row)
    per_chain = []
    for chain, rows in sorted(chains.items()):
        distances = [row['o3p_A'] for row in rows if row['o3p_A'] is not None]
        per_chain.append(dict(chain_index=chain, expected_links=len(rows),
                              missing_links=sum(not row['bond_present'] for row in rows),
                              unmeasured_links=len(rows) - len(distances),
                              short_links=sum(d < low for d in distances),
                              long_links=sum(d > high for d in distances),
                              min_o3p_A=min(distances, default=None),
                              max_o3p_A=max(distances, default=None)))
    reasons = []
    if metrics['missing_links']:
        reasons.append('missing_topology_links')
    if not per_chain or any(c['unmeasured_links'] for c in per_chain):
        reasons.append('unmeasured_link_distances')
    if any(c['short_links'] or c['long_links'] for c in per_chain):
        reasons.append('out_of_screen_link_length')
    if any(row[field] is None for row in metrics['links']
           for field in ('c3_o3_p_deg', 'o3_p_o5_deg')):
        reasons.append('unmeasured_link_angles')
    if metrics['short_heavy_contacts']:
        reasons.append('short_heavy_atom_contacts')
    if metrics['atoms_without_element']:
        reasons.append('atoms_without_element')
    return dict(status='failed' if reasons else 'passed_triage_only',
                failures=reasons, o3p_screen_A=O3P_SCREEN_A, per_chain=per_chain)


def controls(seed=SEED):
    rng = np.random.default_rng(seed)
    steps = np.column_stack([rng.uniform(0.9, 1.2, 11), rng.normal(0, 0.12, (11, 2))])
    steps[4:7] += [[0, 1.6, 0.7], [0, -2.0, -0.55], [0, 0.75, 0.3]]
    return np.vstack([np.zeros(3), np.cumsum(steps, axis=0)])


def benchmark(curve_only, *, preserve_endpoints=False, max_step_deviation_nm=None, seed=SEED):
    geom = runpy.run_path(str(ROOT / 'mdna' / 'path_geometry.py'))
    points = controls(seed)
    updated, metrics = geom['regularize_open_path'](points, max_deviation=BUDGET_NM)
    report = dict(seed=seed, budget_nm=BUDGET_NM, sequence=SEQUENCE, input_points_nm=points.tolist(),
                  smoothed_points_nm=updated.tolist(), curve=metrics,
                  endpoint_displacement_nm=np.linalg.norm((updated-points)[[0,-1]], axis=1).tolist(),
                  atom_level=None)
    resolution_check = {}
    for count in (48, 161, 401):
        baseline, _ = geom['_sample'](points, count)
        smoothed, _ = geom['_sample'](updated, count)
        resolution_check[str(count)] = dict(
            max_turn_before_deg=float(geom['_turns'](baseline).max()),
            max_turn_after_deg=float(geom['_turns'](smoothed).max()),
            max_paired_deviation_nm=float(np.linalg.norm(baseline-smoothed, axis=1).max()),
        )
    report['resolution_check'] = resolution_check
    if not curve_only:
        sys.path.insert(0, str(ROOT))
        from mdna import make
        from mdna.path_geometry import StepSpacingError
        from mdna.structure_qc import measure_open_chain_geometry
        straight = np.linspace(points[0], points[-1], len(points))
        structures = {}
        origins = {}
        cases = [('straight', straight), ('original', points), ('regularized', points)]
        if preserve_endpoints:
            cases += [('endpoint_original', points), ('endpoint_regularized', points)]
        if max_step_deviation_nm is not None:
            cases.append(('endpoint_step_limited_regularized', points))
        for name, cp in cases:
            anchored = name.startswith('endpoint_')
            spacing_limit = max_step_deviation_nm if name == 'endpoint_step_limited_regularized' else None
            try:
                model = make(sequence=SEQUENCE, control_points=cp,
                             path_deviation_nm=BUDGET_NM if name.endswith('regularized') else None,
                             preserve_path_endpoints=anchored, max_step_deviation_nm=spacing_limit)
            except StepSpacingError as exc:
                structures[name] = dict(status='rejected_step_spacing', detail=str(exc),
                                        requested_step_deviation_nm=spacing_limit)
                continue
            origins[name] = model.get_frames()[:, 0, 0, :]
            structures[name] = measure_open_chain_geometry(model.get_traj())
            structures[name]['max_frame_turn_deg'] = float(geom['_turns'](origins[name]).max())
            structures[name]['triage'] = screen_atomic_geometry(structures[name])
            if anchored:
                distances = np.linalg.norm(np.diff(origins[name], axis=0), axis=1)
                structures[name]['frame_step_range_nm'] = [float(distances.min()), float(distances.max())]
                structures[name]['max_frame_step_deviation_nm'] = float(np.abs(distances-.34).max())
                structures[name]['endpoint_error_nm'] = np.linalg.norm(
                    origins[name][[0, -1]]-cp[[0, -1]], axis=1).tolist()
                structures[name]['max_paired_frame_shift_nm'] = float(np.linalg.norm(
                    origins[name]-origins['endpoint_original'], axis=1).max())
                if name == 'endpoint_step_limited_regularized':
                    structures[name]['requested_step_deviation_nm'] = max_step_deviation_nm
        report['atom_level'] = structures
        shift = np.linalg.norm(origins['regularized'] - origins['original'], axis=1)
        report['post_rescale_max_frame_shift_nm'] = float(shift.max())
        report['post_rescale_endpoint_shift_nm'] = shift[[0, -1]].tolist()
        report['post_rescale_exceeds_input_budget'] = bool(shift.max() > BUDGET_NM)
    return report


def plot_curve_comparison(report, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    geom = runpy.run_path(str(ROOT / 'mdna' / 'path_geometry.py'))
    fig = plt.figure(figsize=(10, 4.1), layout='constrained')
    fig.suptitle(f"MDNA: synthetic 3D path | seed {report['seed']}", fontweight='bold')
    grid = fig.add_gridspec(2, 2)
    xy = fig.add_subplot(grid[0, 0])
    xz = fig.add_subplot(grid[1, 0], sharex=xy)
    turn_ax = fig.add_subplot(grid[:, 1])
    count = report['curve']['samples']
    for key, label, color in [('input_points_nm', 'Original', '#A64228'),
                              ('smoothed_points_nm', 'Adjusted', '#147879')]:
        points = np.array(report[key])
        curve, _ = geom['_sample'](points, 401)
        for ax, axis in ((xy, 1), (xz, 2)):
            ax.plot(curve[:, 0], curve[:, axis], color=color, linewidth=1.7)
            ax.scatter(points[:, 0], points[:, axis], color=color, s=10)
        sampled, _ = geom['_sample'](points, count)
        turn_ax.plot(np.linspace(0, 100, count)[1:-1], geom['_turns'](sampled),
                     color=color, label=label, linewidth=1.7)
    xy.set(ylabel='Y (nm)', title='Open path and controls')
    xy.tick_params(labelbottom=False)
    xz.set(xlabel='X (nm)', ylabel='Z (nm)')
    for ax in (xy, xz):
        ax.grid(alpha=.18)
    turn_ax.set(xlabel='Fractional arc length (%)', ylabel='Local turn (degrees)', title=f'{count} arc length samples')
    turn_ax.legend(frameon=False)
    turn_ax.grid(alpha=.18)
    fig.savefig(output, dpi=170)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curve-only', action='store_true')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--preserve-endpoints', action='store_true',
                        help='Include fixed-endpoint atomic comparisons.')
    parser.add_argument('--max-step-deviation-nm', type=float,
                        help='Add a fixed-endpoint smoothing case with this maximum frame-step deviation.')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plot', type=Path)
    args = parser.parse_args()
    if args.max_step_deviation_nm is not None and (not args.preserve_endpoints or args.curve_only):
        parser.error('--max-step-deviation-nm requires the fixed-endpoint atomic comparison')
    with redirect_stdout(sys.stderr):
        report = benchmark(args.curve_only, preserve_endpoints=args.preserve_endpoints,
                           max_step_deviation_nm=args.max_step_deviation_nm, seed=args.seed)
        if args.plot:
            plot_curve_comparison(report, args.plot)
    text = json.dumps(report, allow_nan=False, indent=2) + '\n'
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
