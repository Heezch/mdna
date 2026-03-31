# Installation

## Quick Install (pip)

```bash
pip install mdna
```

## From Source

Clone the repository:

```bash
git clone git@github.com:heezch/mdna.git
cd mdna
pip install -e .
```

## Dependencies

MDNA requires Python 3.10+ and depends on:

| Package | Purpose |
|---------|---------|
| **numpy** | Numerical arrays and linear algebra |
| **scipy** | Spline interpolation, rotations |
| **mdtraj** | Molecular trajectory I/O and topology |
| **matplotlib** | Visualization and plotting |
| **numba** | JIT-compiled geometry routines |
| **tables** | HDF5 I/O for atomic base templates (PyTables) |
| **joblib** | Parallel computation in groove analysis |

### Optional Dependencies

| Package | Purpose |
|---------|---------|
| **openmm** | Molecular dynamics simulation via `sequence_to_md()` |
| **nglview** | Interactive 3D molecular visualization in notebooks |
| **Cython** | Build fast PyLk extensions for writhe/linking number |

## Verify Installation

```python
import mdna
print(mdna.__version__)

# Quick smoke test — generate a 12-bp DNA
dna = mdna.make(sequence='CGCGAATTCGCG')
dna.describe()
```

If this prints the DNA structure info without errors, you're ready to go.

## Building PyLk Cython Extensions

MDNA bundles `PyLk` (at `mdna/simulate/Evals/PyLk`) for linking-number and writhe calculations. Without the compiled Cython extensions you may see warnings like *"Cython version of writhemap/linkingnumber not compiled"*. The pure-Python and Numba fallbacks still work, but the compiled Cython version is significantly faster.

### 1. Install Cython

```bash
# conda / mamba
mamba install -n mdna cython

# or pip
pip install Cython
```

### 2. Compile in-place

```bash
# from the repository root
cd mdna/simulate/Evals/PyLk
python setup.py build_ext --inplace
```

If you use conda/mamba, you can run with your environment explicitly:

```bash
mamba run -n mdna python mdna/simulate/Evals/PyLk/setup.py build_ext --inplace
```

### 3. Verify

```python
from mdna.simulate.Evals.PyLk.pylk import writhemap, linkingnumber
print("PyLk OK:", writhemap.__name__, linkingnumber.__name__)
```

## Filament Dataset (for notebooks)

Several Jupyter notebooks (e.g. the filament tutorial) require an external dataset hosted on Figshare. To download it:

```bash
cd docs/notebooks
export MDNA_FILAMENT_DATASET_URL='https://doi.org/10.6084/m9.figshare.31423193'
python ./scripts/fetch_filament_dataset.py --output-root ./data
```

The fetch script resolves the DOI to the direct download URL, downloads the archive, and unpacks it into `data/filament_dataset/`.

!!! tip "Optional integrity check"
    If you have the published SHA256 hash of the archive, you can verify:

    ```bash
    export MDNA_FILAMENT_DATASET_SHA256='<REAL_SHA256>'
    python ./scripts/fetch_filament_dataset.py --output-root ./data --force
    ```

    If you do not have the hash, leave `MDNA_FILAMENT_DATASET_SHA256` unset and run without checksum verification.
