# H-NS Filament Dataset

This folder contains MD trajectories used to build and test H-NS filament assemblies in the filament tutorial.

## What is in this dataset?

- **`0_s1s1/`**: trajectories for the H-NS **s1-s1** dimerization system
- **`1_s2s2/`**: trajectories for the H-NS **s2-s2** multimerization system
- **`FI/`**: trajectories for the H-NS **DBD-DNA complex** (used to place DNA)

## Quick inventory

- `0_s1s1/`
  - 18 trajectory pairs (`.xtc` + `.pdb`): `dry_0` to `dry_15`, plus `dry_closed`, `dry_open`
- `1_s2s2/`
  - 10 trajectory pairs (`.xtc` + `.pdb`): `dry_0` to `dry_9`
- `FI/`
  - 16 trajectory pairs (`.xtc` + `.pdb`): `dry_0` to `dry_15`
  - plus `FI_HNS_highaff.pdb`

## Size (on disk)

- Full dataset: ~`1.7G`
- `0_s1s1/`: ~`970M`
- `1_s2s2/`: ~`342M`
- `FI/`: ~`416M`

## Biological/structural context (accessible summary)

H-NS is a bacterial DNA-organizing protein. It has:

- an **oligomerization region** (residues 1-83), containing:
  - **s1**: homodimerization site
  - **s2**: multimerization site
- a **DNA-binding domain (DBD)** (residues 89-137)
- a flexible linker between these parts

In this project, filament assembly is done by identifying functional segments (`s1`, `h3`, `s2`, `l2`, `dbd`) and superimposing overlapping residues (SVD/RMSD-based fitting) to build continuous filaments.

## Simulation metadata (source protocol summary)

- **s1s1 system**
  - 15 runs of 62.5 ns from same start
  - 2 longer runs of 100 ns (`open` and `closed` DBD availability states)
- **s2s2 system**
  - 12 runs of 100 ns from same start
- **General MD setup**
  - GROMACS 2020.4
  - AMBER14sb-parmbsc1 + TIP3P water
  - periodic dodecahedron box (>=1.0 nm margin)
  - 50 mM NaCl
  - PME electrostatics, 1.1 nm nonbonded cutoff
  - NPT at 298 K and 1 bar
  - 2 fs timestep, leap-frog integrator

## How this folder is used in the examples

In `examples/4_filament_tutorial.ipynb` (full-data mode), these trajectories are loaded to build `site_map` and then assemble filaments. For small-package distribution, use the separate reduced dataset in `examples/data/filament_minimal/`.
