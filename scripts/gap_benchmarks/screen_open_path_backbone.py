"""Evaluate a finite set of fixed-endpoint DNA paths against backbone distances."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

import numpy as np

from evaluate_open_path_bends import BUDGET_NM, ROOT, SEQUENCE, controls, screen_atomic_geometry


SHAPE_BUDGETS_NM = (0.15, 0.30, 0.45, 0.60, 0.75, BUDGET_NM)


def evaluate(seed, step_limit):
    sys.path.insert(0, str(ROOT))
    from mdna import make
    from mdna.path_geometry import StepSpacingError, _turns
    from mdna.structure_qc import measure_open_chain_geometry

    points = controls(seed)

    def measure(model):
        origins = model.get_frames()[:, 0, 0, :]
        links = measure_open_chain_geometry(model.get_traj())
        triage = screen_atomic_geometry(links)
        steps = np.linalg.norm(np.diff(origins, axis=0), axis=1)
        return dict(max_turn_deg=float(_turns(origins).max()),
                    short_links=sum(c["short_links"] for c in triage["per_chain"]),
                    long_links=sum(c["long_links"] for c in triage["per_chain"]),
                    short_heavy_contacts=links["short_heavy_contacts"],
                    missing_links=links["missing_links"],
                    unmeasured_links=sum(c["unmeasured_links"] for c in triage["per_chain"]),
                    triage_status=triage["status"],
                    max_step_deviation_nm=float(np.abs(steps-.34).max()),
                    endpoint_error_nm=np.linalg.norm(origins[[0, -1]]-points[[0, -1]], axis=1).tolist()), origins

    baseline, reference = measure(make(sequence=SEQUENCE, control_points=points,
                                       preserve_path_endpoints=True))
    cases = []
    accepted = []
    for budget in SHAPE_BUDGETS_NM:
        try:
            model = make(sequence=SEQUENCE, control_points=points, preserve_path_endpoints=True,
                         path_deviation_nm=budget, max_step_deviation_nm=step_limit)
        except StepSpacingError as exc:
            cases.append(dict(search_budget_nm=budget, status="rejected_step_spacing", detail=str(exc)))
            continue
        observed, origins = measure(model)
        shift = float(np.linalg.norm(origins-reference, axis=1).max())
        reasons = [key for key in ("short_links", "long_links", "short_heavy_contacts",
                                  "missing_links", "unmeasured_links") if observed[key] > baseline[key]]
        if observed["max_turn_deg"] >= baseline["max_turn_deg"] - 1e-9:
            reasons.append("no_turn_improvement")
        if any(observed["endpoint_error_nm"]) or shift > BUDGET_NM:
            reasons.append("path_fidelity")
        candidate = dict(search_budget_nm=budget, max_paired_frame_shift_nm=shift,
                         failures=reasons, **observed)
        cases.append(candidate)
        if not reasons:
            accepted.append(candidate)
    best = min(accepted, key=lambda row: (row["max_turn_deg"], row["search_budget_nm"]), default=None)
    return dict(seed=seed, baseline=baseline, candidates=cases, selected=best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, nargs="+", required=True)
    parser.add_argument("--max-step-deviation-nm", type=float, default=.065)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        results = [evaluate(seed, args.max_step_deviation_nm) for seed in args.seed]
    report = dict(sequence=SEQUENCE, path_budget_nm=BUDGET_NM,
                  spacing_deviation_limit_nm=args.max_step_deviation_nm,
                  search_budgets_nm=SHAPE_BUDGETS_NM, results=results)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
