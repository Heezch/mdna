"""Relax standard DNA backbone with fixed base heavy atoms and termini."""

import argparse
from contextlib import redirect_stdout
from hashlib import sha256
import json
from pathlib import Path
import random
import sys

import numpy as np

from audit_resampled_orientation_torsions import compare, references
from evaluate_open_path_bends import ROOT, controls, screen_atomic_geometry
from probe_amber_anchor_restraints import _glycosidic_base_group
from probe_open_path_resampling import frames_on_grid
from probe_resampled_orientation import angle_outliers, rotate_frames


def replay(source):
    from mdna.generators import StructureGenerator

    raw = source.read_bytes()
    report = json.loads(raw)
    centers_raw = source.with_name(report["source_report"]).read_bytes()
    if sha256(centers_raw).hexdigest() != report["source_sha256"]:
        raise ValueError("Frozen center report hash differs")
    frozen = json.loads(centers_raw)
    frames = frames_on_grid(controls(frozen["seed"]), np.asarray(frozen["final"]["frame_origins_nm"]))
    sweep, = [row for row in report["sweeps"] if row["torsion_cap_deg"] == 60.]
    for step in sweep["steps"]:
        if step["status"] == "accepted":
            frames = rotate_frames(frames, step["frames"], step["local_basis_axis"], step["angles_deg"])
    generator = StructureGenerator(sequence=frozen["sequence"], frames=frames, circular=False)
    return frozen["seed"], sha256(raw).hexdigest(), generator.get_traj(remove_terminal_phosphates=True)


def relax(trajectory, strengths, iterations, panels):
    from openmm import CustomExternalForce, LocalEnergyMinimizer, Platform, VerletIntegrator, unit
    from openmm.app import ForceField, Modeller, NoCutoff
    from mdna.structure_qc import measure_open_chain_geometry

    heavy = [atom.index for atom in trajectory.topology.atoms if atom.element.atomic_number > 1]
    trajectory = trajectory.atom_slice(heavy)
    modeller = Modeller(trajectory.topology.to_openmm(), trajectory.xyz[0].astype(float) * unit.nanometer)
    forcefield = ForceField("amber14-all.xml", "implicit/gbn2.xml")
    cpu = Platform.getPlatformByName("CPU")
    cpu.setPropertyDefaultValue("Threads", "2")
    state = random.getstate()
    try:
        random.seed(0)
        modeller.addHydrogens(forcefield, pH=7.0, platform=cpu)
    finally:
        random.setstate(state)
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=NoCutoff, constraints=None)
    prepared_xyz = modeller.positions.value_in_unit(unit.nanometer)
    prepared_xyz = np.asarray(prepared_xyz)
    heavy_indexes = [atom.index for atom in modeller.topology.atoms() if atom.element.symbol != "H"]
    def identities(atoms):
        return [(atom.residue.index, atom.residue.name, atom.name) for atom in atoms]

    if (identities(atom for atom in modeller.topology.atoms() if atom.element.symbol != "H")
            != identities(trajectory.topology.atoms)
            or not np.array_equal(prepared_xyz[heavy_indexes], trajectory.xyz[0])):
        raise ValueError("Preparing the Amber system changed the input heavy atom mapping or coordinates")

    fixed = set()
    for chain in modeller.topology.chains():
        residues = list(chain.residues())
        for residue in residues:
            base, _ = _glycosidic_base_group(modeller.topology, residue)
            fixed.update(base)
            fixed.update(atom.index for atom in residue.atoms()
                         if atom.name == "C1'" or residue in (residues[0], residues[-1]))
    for index in fixed:
        system.setParticleMass(index, 0. * unit.dalton)
    mobile_heavy = sorted(set(heavy_indexes) - fixed)
    restraints = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraints.addGlobalParameter("k", 0.)
    for name in ("x0", "y0", "z0"):
        restraints.addPerParticleParameter(name)
    for index in mobile_heavy:
        restraints.addParticle(index, prepared_xyz[index].tolist())
    restraints.setForceGroup(1)
    system.addForce(restraints)
    integrator = VerletIntegrator(1. * unit.femtoseconds)
    from openmm import Context
    context = Context(system, integrator, cpu)

    def inspect(xyz):
        trajectory.xyz[0] = xyz[heavy_indexes]
        qc = measure_open_chain_geometry(trajectory)
        return qc, dict(triage=screen_atomic_geometry(qc), angle_outliers=angle_outliers(qc["links"]),
                        short_heavy_contacts=qc["short_heavy_contacts"])

    initial_qc, initial = inspect(prepared_xyz)
    results = []
    for strength in strengths:
        context.setParameter("k", strength)
        context.setPositions(modeller.positions)
        before = context.getState(getEnergy=True, groups={0}).getPotentialEnergy()
        LocalEnergyMinimizer.minimize(context, 10. * unit.kilojoule_per_mole / unit.nanometer, iterations)
        state = context.getState(getEnergy=True, getPositions=True, groups={0})
        final_xyz = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        qc, geometry = inspect(final_xyz)
        displacements = np.linalg.norm(final_xyz - prepared_xyz, axis=1) * 10.
        if np.max(displacements[list(fixed)]) > 1e-8:
            raise RuntimeError("Fixed nuclei moved during minimization")
        results.append(dict(restraint_k_kJ_mol_nm2=strength, geometry=geometry,
                            energy_kJ_mol=[before.value_in_unit(unit.kilojoule_per_mole),
                                           state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)],
                            max_fixed_shift_A=float(np.max(displacements[list(fixed)])),
                            max_mobile_heavy_shift_A=float(np.max(displacements[mobile_heavy])),
                            rms_mobile_heavy_shift_A=float(np.sqrt(np.mean(displacements[mobile_heavy]**2))),
                            reference_torsions={entry: compare(initial_qc["links"], qc["links"], source["by_step"], set())["all_links"]
                                                for entry, source in panels.items()},
                            heavy_xyz_nm=final_xyz[heavy_indexes].tolist()))
    return dict(initial=initial, atom_keys=identities(trajectory.topology.atoms),
                initial_heavy_xyz_nm=prepared_xyz[heavy_indexes].tolist(),
                system_particles=system.getNumParticles(), fixed_particles=len(fixed),
                mobile_heavy_atoms=len(mobile_heavy), results=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--strength", type=float, nargs="+", default=(10000., 1000.))
    parser.add_argument("--iterations", type=int, default=120)
    args = parser.parse_args()
    if args.iterations < 1 or any(not np.isfinite(k) or k <= 0. for k in args.strength):
        parser.error("Use a positive iteration limit and finite positive restraint strengths.")
    sys.path.insert(0, str(ROOT))
    with redirect_stdout(sys.stderr):
        seed, source_hash, trajectory = replay(args.source)
        panels = references(args.reference)
        report = relax(trajectory, args.strength, args.iterations, panels)
    report.update(seed=seed, source_sha256=source_hash, source_report=args.source.name,
                  reference_sha256=sha256(args.reference.read_bytes()).hexdigest(),
                  forcefield=["amber14-all.xml", "implicit/gbn2.xml"],
                  platform="CPU", iterations=args.iterations)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
