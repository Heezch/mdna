# mdna.simulate — Integrated Monte Carlo Simulation Engine

This subpackage provides the Polymer Monte Carlo (PMC) simulation engine used by MDNA for energy minimisation and conformational sampling of double-stranded DNA. It is a fully integrated adaptation of [PMCpy](https://github.com/eskoruppa/PMCpy) by Enrico Skoruppa, vendored directly into the MDNA package so that no external submodules or separate installation steps are required.

---

## Origin & Integration

The original **PMCpy** package implements sequence-dependent, coarse-grained rigid base pair-step models for efficient Monte Carlo sampling of DNA conformations. MDNA integrates this code as `mdna.simulate` with the following adaptations:

- **Namespace**: all `pmcpy.*` imports are replaced with `mdna.simulate.*`.
- **SO3 / SE(3) math**: the upstream `SO3` git submodule is replaced by `mdna.math`, which provides the same rotation and rigid-body routines.
- **Conditional JIT**: the upstream `pyConDec` conditional-decorator submodule is replaced by `mdna._jit.cond_jit`, which wraps Numba's `@njit` when Numba is available and falls back to a no-op decorator otherwise.
- **PyLk**: the linking-number / writhe library is bundled at `mdna/simulate/Evals/PyLk` (no submodule).
- **Stiffness data**: sequence-dependent stiffness matrices (`RBPStiff`) are bundled in `mdna/simulate/BPStep/RBPStiff/`.
- **`aux.py` → `utils.py`**: renamed to follow upstream's March 2026 refactor, with `@cond_jit` applied to all JIT-decorated functions.

Apart from these integration changes, the simulation logic, Monte Carlo moves, and energy model are kept faithful to the upstream implementation.

---

## Features

| Feature | Description |
|---------|-------------|
| **Sequence-dependent energy** | Base pair-step stiffness and ground-state parameters from atomistic MD (Lankas / RBP parametrisation) |
| **Flexible boundaries** | Open linear chains and closed (circular) DNA |
| **Monte Carlo moves** | Pivot, Double Pivot, Crankshaft, Cluster Translation, Single Triad, Midstep Move |
| **Excluded volume** | Bead-based EV interactions with self-crossing detection |
| **Constraints** | Stretching forces (tweezer geometry), repulsion planes, fixed triads |
| **Equilibration** | Automated convergence detection for burn-in |
| **Observables** | Tangent-tangent correlation, persistence length, writhe, linking number |
| **Trajectory I/O** | XYZ-format trajectory writing and reading |
| **Numba acceleration** | Optional JIT compilation via `cond_jit(nopython=True, cache=True)` |

---

## Package Structure

```
mdna/simulate/
├── __init__.py          # Public API re-exports
├── chain.py             # Chain class — manages triads, positions, conf
├── pmc.py               # Core PMC loop
├── utils.py             # Utility functions (params2conf, triad_realign, etc.)
├── BPStep/              # Base pair step energy (BPStep, RBP, stiffness data)
├── BlockMat/            # Block-diagonal matrix utilities
├── Constraints/         # External constraints (repulsion planes, etc.)
├── Dumps/               # Trajectory I/O (XYZ format)
├── Evals/               # Observables (tangent correlation, PyLk)
├── ExVol/               # Excluded volume interactions
├── GenConfs/            # Configuration generators (straight, from sequence)
├── MCStep/              # Monte Carlo move implementations
├── apps/                # Application-level scripts (nucleosome free energy, etc.)
└── run/                 # High-level Run class and equilibration protocol
```

---

## Usage within MDNA

Most users interact with the simulation engine indirectly through the `Minimizer` class:

```python
import mdna

dna = mdna.make(sequence="A" * 100, circular=True)
dna.minimize()
```

For direct access to the simulation engine:

```python
from mdna.simulate import Run
from mdna.simulate.GenConfs.straight import gen_straight
import numpy as np

nbp = 500
sequence = "".join(np.random.choice(list("ATCG"), nbp))
triads, positions = gen_straight(nbp)

sim = Run(
    triads=triads,
    positions=positions,
    sequence=sequence,
    closed=False,
    endpoints_fixed=True,
    temp=300,
    exvol_rad=2.0,
    parameter_set="md",
)

sim.run(num_steps=100_000, dump_every=1_000, outfile="traj.xyz")
```

---

## Citation

The simulation engine is based on PMCpy. If you use Monte Carlo simulation features in your research, please cite:

> Enrico Skoruppa, Helmut Schiessel,
> **Systematic coarse-graining of sequence-dependent structure and elasticity of double-stranded DNA**,
> *Physical Review Research* **7**, 013044 (2025).
> DOI: [10.1103/PhysRevResearch.7.013044](https://doi.org/10.1103/PhysRevResearch.7.013044)

> Willem Vanderlinden, Enrico Skoruppa, Pauline J. Kolbeck, Enrico Carlon, Jan Lipfert,
> **DNA fluctuations reveal the size and dynamics of topological domains**,
> *PNAS Nexus* **1**(5), pgac268 (2022).
> DOI: [10.1093/pnasnexus/pgac268](https://doi.org/10.1093/pnasnexus/pgac268)

---

## License

The original PMCpy is released under the [GNU General Public License v2.0](https://github.com/eskoruppa/PMCpy/blob/main/LICENSE). The integrated code in `mdna.simulate` retains this license for the vendored portions.