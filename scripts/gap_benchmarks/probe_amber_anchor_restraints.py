"""Probe Amber14 local DNA minimization with all non-target atoms fixed.

This is an environment and restraint-semantics check, not a gap-closure
benchmark. It loads a standard DNA PDB, adds hydrogens, fixes every atom
outside one selected residue by setting particle masses to zero, minimizes,
and reports energy change and maximum fixed-coordinate drift. It writes no
structure or other output file.

Run inside an environment that already provides OpenMM and MDTraj, for example
the project's existing WSL ``md-env`` Conda environment::

    python scripts/gap_benchmarks/probe_amber_anchor_restraints.py \
      artifacts/gap_benchmarks/119d_source.pdb
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


DNA_RESIDUES = frozenset({"DA", "DC", "DG", "DT"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _glycosidic_base_group(topology: Any, residue: Any) -> tuple[list[int], list[int]]:
    """Find the base-side component after cutting C1'-N1/N9 in one residue."""
    residue_atoms = [atom for atom in topology.atoms() if atom.residue == residue]
    atoms_by_name = {atom.name: atom for atom in residue_atoms}
    glycosidic_name = "N1" if residue.name in {"DC", "DT"} else "N9"
    if "C1'" not in atoms_by_name or glycosidic_name not in atoms_by_name:
        raise ValueError(f"Cannot identify the glycosidic bond in {residue.name} {residue.id}")
    glyco_n = atoms_by_name[glycosidic_name]
    adjacency: dict[int, set[int]] = {atom.index: set() for atom in residue_atoms}
    for bond in topology.bonds():
        first, second = bond.atom1, bond.atom2
        if first.residue != residue or second.residue != residue:
            continue
        if {first.name, second.name} == {"C1'", glycosidic_name}:
            continue
        adjacency[first.index].add(second.index)
        adjacency[second.index].add(first.index)
    visited = {glyco_n.index}
    stack = [glyco_n.index]
    while stack:
        current = stack.pop()
        for neighbor in adjacency[current] - visited:
            visited.add(neighbor)
            stack.append(neighbor)
    group = [atom for atom in residue_atoms if atom.index in visited]
    heavy = [atom.index for atom in group if atom.element.symbol != "H"]
    if not heavy:
        raise ValueError(f"No base heavy atoms found in {residue.name} {residue.id}")
    return [atom.index for atom in group], heavy


def run_probe(
    pdb_path: Path,
    *,
    chain_id: str = "A",
    mobile_residue_index: int | None = None,
    max_iterations: int = 0,
    platform_name: str = "Reference",
    hydrogen_seed: int = 0,
    constraints: str = "none",
    perturb_glycosidic_degrees: float = 0.0,
) -> dict[str, Any]:
    """Minimize one DNA residue while keeping every other atom exactly fixed."""
    try:
        import numpy as np
        from openmm import LocalEnergyMinimizer, MinimizationReporter, Platform, VerletIntegrator, unit
        from openmm.app import ForceField, HBonds, Modeller, NoCutoff, PDBFile, Simulation
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("This probe requires OpenMM and its application module.") from exc

    pdb_path = pdb_path.resolve()
    if not pdb_path.is_file():
        raise FileNotFoundError(pdb_path)
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative; zero means no iteration cap")
    if constraints not in {"none", "hbonds"}:
        raise ValueError("constraints must be 'none' or 'hbonds'")

    pdb = PDBFile(str(pdb_path))
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.delete([residue for residue in modeller.topology.residues()
                     if residue.name not in DNA_RESIDUES])
    chain_residues = [residue for residue in modeller.topology.residues()
                      if residue.chain.id == chain_id and residue.name in DNA_RESIDUES]
    if len(chain_residues) < 3:
        raise ValueError(f"Expected at least three standard DNA residues in chain {chain_id!r}")
    if mobile_residue_index is None:
        selected_residue = chain_residues[len(chain_residues) // 2]
    else:
        if not 0 <= mobile_residue_index < len(chain_residues):
            raise ValueError("mobile_residue_index is outside the selected DNA chain")
        selected_residue = chain_residues[mobile_residue_index]
    mobile_key = (selected_residue.chain.id, selected_residue.id)

    forcefield = ForceField("amber14-all.xml", "implicit/gbn2.xml")
    python_rng_state = random.getstate()
    try:
        random.seed(hydrogen_seed)
        modeller.addHydrogens(forcefield, pH=7.0)
    finally:
        random.setstate(python_rng_state)
    # addHydrogens rebuilds Modeller's topology, so do not retain Residue object
    # identity across that operation.
    mobile = next(residue for residue in modeller.topology.residues()
                  if (residue.chain.id, residue.id) == mobile_key)
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=NoCutoff,
        constraints=HBonds if constraints == "hbonds" else None,
    )
    mobile_atom_indices = {atom.index for atom in modeller.topology.atoms()
                           if atom.residue == mobile}
    fixed_atom_indices = [atom.index for atom in modeller.topology.atoms()
                          if atom.index not in mobile_atom_indices]
    for atom_index in fixed_atom_indices:
        system.setParticleMass(atom_index, 0 * unit.dalton)

    integrator = VerletIntegrator(1.0 * unit.femtoseconds)
    platform = Platform.getPlatformByName(platform_name)
    simulation = Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    initial = simulation.context.getState(getEnergy=True, getPositions=True)
    initial_xyz = initial.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    class Reporter(MinimizationReporter):
        def __init__(self) -> None:
            super().__init__()
            self.last: dict[str, Any] | None = None
            self.report_count = 0
            self.restart_count = 0
            self.previous_iteration: int | None = None

        def report(self, iteration, x, grad, args):
            if self.previous_iteration is not None and iteration <= self.previous_iteration:
                self.restart_count += 1
            self.previous_iteration = int(iteration)
            gradient = np.asarray(grad, dtype=float)
            self.last = {
                "iteration": int(iteration),
                "system_energy_kJ_mol": float(args["system energy"]),
                "restraint_energy_kJ_mol": float(args["restraint energy"]),
                "restraint_strength_kJ_mol_nm2": float(args["restraint strength"]),
                "max_constraint_error_fraction": float(args["max constraint error"]),
                "objective_gradient_rms_kJ_mol_nm": float(np.sqrt(np.mean(gradient**2))),
                "objective_gradient_max_abs_kJ_mol_nm": float(np.max(np.abs(gradient), initial=0.0)),
            }
            self.report_count += 1
            return False

    def minimize() -> Reporter:
        current = Reporter()
        LocalEnergyMinimizer.minimize(
            simulation.context,
            tolerance=10 * unit.kilojoule_per_mole / unit.nanometer,
            maxIterations=max_iterations,
            reporter=current,
        )
        return current

    reporter = minimize()
    baseline = simulation.context.getState(getEnergy=True, getPositions=True)
    baseline_xyz = baseline.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    baseline_energy = baseline.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    minimization_start_energy = initial.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    perturbation_report = None
    if perturb_glycosidic_degrees != 0.0:
        group_indices, base_heavy_indices = _glycosidic_base_group(modeller.topology, mobile)
        atom_by_name = {atom.name: atom for atom in modeller.topology.atoms()
                        if atom.residue == mobile}
        glycosidic_name = "N1" if mobile.name in {"DC", "DT"} else "N9"
        origin = baseline_xyz[atom_by_name["C1'"].index]
        axis = baseline_xyz[atom_by_name[glycosidic_name].index] - origin
        axis = axis / np.linalg.norm(axis)
        theta = np.radians(perturb_glycosidic_degrees)
        cosine, sine = np.cos(theta), np.sin(theta)
        perturbed_xyz = baseline_xyz.copy()
        vectors = perturbed_xyz[group_indices] - origin
        perturbed_xyz[group_indices] = (
            vectors * cosine
            + np.cross(axis, vectors) * sine
            + np.outer(vectors @ axis, axis) * (1.0 - cosine)
            + origin
        )
        before_rmsd = float(np.sqrt(np.mean(np.sum(
            (perturbed_xyz[base_heavy_indices] - baseline_xyz[base_heavy_indices]) ** 2,
            axis=1,
        ))))
        simulation.context.setPositions(perturbed_xyz * unit.nanometer)
        perturbed_initial = simulation.context.getState(getEnergy=True)
        minimization_start_energy = perturbed_initial.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )
        reporter = minimize()
        final = simulation.context.getState(getEnergy=True, getPositions=True)
        final_xyz = final.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        after_rmsd = float(np.sqrt(np.mean(np.sum(
            (final_xyz[base_heavy_indices] - baseline_xyz[base_heavy_indices]) ** 2,
            axis=1,
        ))))
        perturbation_report = {
            "perturbation_degrees": perturb_glycosidic_degrees,
            "base_group_particle_count": len(group_indices),
            "base_group_heavy_atom_count": len(base_heavy_indices),
            "pre_minimization_base_rmsd_from_baseline_A": before_rmsd * 10.0,
            "post_minimization_base_rmsd_from_baseline_A": after_rmsd * 10.0,
            "base_rmsd_recovery_fraction": (
                1.0 - after_rmsd / before_rmsd if before_rmsd else 1.0
            ),
            "perturbed_initial_energy_kJ_mol": float(
                minimization_start_energy
            ),
            "final_energy_kJ_mol": float(final.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)),
        }
    else:
        final = baseline
    final_xyz = final.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    drift_nm = np.linalg.norm(final_xyz[fixed_atom_indices] - initial_xyz[fixed_atom_indices], axis=1)
    mobile_heavy_indices = [atom.index for atom in modeller.topology.atoms()
                            if atom.index in mobile_atom_indices
                            and atom.element.symbol != "H"]
    mobile_heavy_shift_nm = np.linalg.norm(
        final_xyz[mobile_heavy_indices] - initial_xyz[mobile_heavy_indices], axis=1
    )
    initial_energy = initial.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    final_energy = final.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    constraint_errors = []
    atom_by_index = {atom.index: atom for atom in modeller.topology.atoms()}
    for constraint_index in range(system.getNumConstraints()):
        first, second, target_distance = system.getConstraintParameters(constraint_index)
        target_nm = target_distance.value_in_unit(unit.nanometer)
        observed_nm = float(np.linalg.norm(final_xyz[first] - final_xyz[second]))
        initial_observed_nm = float(np.linalg.norm(initial_xyz[first] - initial_xyz[second]))
        relative_error = abs(observed_nm - target_nm) / target_nm
        initial_relative_error = abs(initial_observed_nm - target_nm) / target_nm
        first_atom = atom_by_index[first]
        second_atom = atom_by_index[second]
        constraint_errors.append({
            "relative_error": relative_error,
            "initial_relative_error": initial_relative_error,
            "atoms": [
                f"{first_atom.residue.chain.id}:{first_atom.residue.id}:{first_atom.name}",
                f"{second_atom.residue.chain.id}:{second_atom.residue.id}:{second_atom.name}",
            ],
            "target_distance_nm": target_nm,
            "initial_distance_nm": initial_observed_nm,
            "final_distance_nm": observed_nm,
            "both_atoms_fixed": first in fixed_atom_indices and second in fixed_atom_indices,
        })
    mobile_masses = [system.getParticleMass(index).value_in_unit(unit.dalton)
                     for index in mobile_atom_indices]
    return {
        "probe": "amber14-gbn2-fixed-nontarget-atoms-v1",
        "source_path": str(pdb_path),
        "source_sha256": _sha256(pdb_path),
        "forcefield": ["amber14-all.xml", "implicit/gbn2.xml"],
        "constraint_policy": constraints,
        "platform": platform_name,
        "hydrogen_seed": hydrogen_seed,
        "chain_id": chain_id,
        "mobile_residue": {"id": mobile.id, "name": mobile.name},
        "added_hydrogens": True,
        "particle_count": system.getNumParticles(),
        "fixed_atom_count": len(fixed_atom_indices),
        "mobile_residue_atom_count": len(mobile_atom_indices),
        "mobile_particle_mass_min_dalton": float(min(mobile_masses)),
        "mobile_particle_mass_max_dalton": float(max(mobile_masses)),
        "constraint_count": system.getNumConstraints(),
        "integrator_constraint_tolerance": float(integrator.getConstraintTolerance()),
        "max_iterations": max_iterations,
        "minimizer_report_count": reporter.report_count,
        "minimizer_restart_count": reporter.restart_count,
        "last_minimizer_report": reporter.last,
        "max_initial_constraint_distance_error_fraction": float(max(
            (item["initial_relative_error"] for item in constraint_errors), default=0.0
        )),
        "max_final_constraint_distance_error_fraction": float(max(
            (item["relative_error"] for item in constraint_errors), default=0.0
        )),
        "worst_final_constraint": max(
            constraint_errors, key=lambda item: item["relative_error"], default=None
        ),
        "initial_energy_kJ_mol": float(initial_energy),
        "baseline_minimized_energy_kJ_mol": float(baseline_energy),
        "final_energy_kJ_mol": float(final_energy),
        "minimization_start_energy_kJ_mol": float(minimization_start_energy),
        "potential_energy_nonincreasing": bool(final_energy <= minimization_start_energy),
        "maximum_fixed_atom_drift_nm": float(drift_nm.max(initial=0.0)),
        "maximum_mobile_heavy_atom_displacement_nm": float(mobile_heavy_shift_nm.max(initial=0.0)),
        "glycosidic_perturbation_recovery": perturbation_report,
        "trajectory_run": False,
        "gap_closure_evaluated": False,
        "scientific_acceptance": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb", type=Path)
    parser.add_argument("--chain", default="A")
    parser.add_argument("--mobile-residue-index", type=int)
    parser.add_argument("--max-iterations", type=int, default=0,
                        help="OpenMM L-BFGS cap; 0 runs until converged")
    parser.add_argument("--platform", default="Reference")
    parser.add_argument("--hydrogen-seed", type=int, default=0)
    parser.add_argument("--constraints", choices=("none", "hbonds"), default="none")
    parser.add_argument("--perturb-glycosidic-degrees", type=float, default=0.0,
                        help="Perturb the base relative to the sugar and measure local recovery")
    args = parser.parse_args()
    print(json.dumps(run_probe(
        args.pdb,
        chain_id=args.chain,
        mobile_residue_index=args.mobile_residue_index,
        max_iterations=args.max_iterations,
        platform_name=args.platform,
        hydrogen_seed=args.hydrogen_seed,
        constraints=args.constraints,
        perturb_glycosidic_degrees=args.perturb_glycosidic_degrees,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
