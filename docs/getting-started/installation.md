# Installation

## Quick Install (pip)

```bash
pip install mdna
```

## From Source

Clone the repository (including submodules for the Monte Carlo engine):

```bash
git clone --recurse-submodules -j8 git@github.com:heezch/mdna.git
cd mdna
pip install -e .
```

## Dependencies

MDNA requires Python 3.9+ and depends on:

| Package | Purpose |
|---------|---------|
| **numpy** | Numerical arrays and linear algebra |
| **scipy** | Spline interpolation, rotations |
| **mdtraj** | Molecular trajectory I/O and topology |
| **matplotlib** | Visualization and plotting |
| **numba** | JIT-compiled geometry routines |

### Optional Dependencies

| Package | Purpose |
|---------|---------|
| **openmm** | Molecular dynamics simulation via `sequence_to_md()` |
| **nglview** | Interactive 3D molecular visualization in notebooks |
| **joblib** | Parallel computation in groove analysis |

## Verify Installation

```python
import mdna
print(mdna.__version__)

# Quick smoke test — generate a 12-bp DNA
dna = mdna.make(sequence='CGCGAATTCGCG')
dna.describe()
```

If this prints the DNA structure info without errors, you're ready to go.
