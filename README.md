# MDNA: a software module for DNA structure generation and analysis

## Description

MDNA is a Python toolkit for building, modifying and analyzing double stranded DNA structures at atomic resolution. It generates arbitrary DNA shapes through spline based mapping, supports canonical and non canonical base modifications, and applies Monte Carlo minimization to obtain physically consistent configurations. The toolkit implements full rigid base analysis, including intra base pair parameters shear, stretch, stagger, buckle, propeller, and opening, as well as inter base pair step parameters shift, slide, rise, tilt, roll, and twist. With built in linking number calculations and seamless interoperability with [MDTraj](https://github.com/mdtraj/mdtraj) and [OpenMM](https://github.com/openmm/openmm), MDNA unifies structure generation, editing, and analysis in a single modular framework for complex DNA and DNA protein assemblies.


## Installation
To install MDNA use pip:
```bash
pip install mdna
```

Or if you want to install the most recent version of MDNA, follow these steps:
```bash
git clone https://github.com/heezch/mdna.git
cd mdna
pip install .
```

## Usage

See our [documentation page](https://heezch.github.io/mdna/)

## Citation

If you use MDNA in your research, please cite:

> Link to the [publication](https://doi.org/10.1093/nar/gkag549) at Nucleic Acids Research, Volume 54, Issue 10, 10 June 2026, gkag549.

If you use the Monte Carlo simulation / minimization features, please also cite the underlying PMCpy work:

> Enrico Skoruppa, Helmut Schiessel,
> **Systematic coarse-graining of sequence-dependent structure and elasticity of double-stranded DNA**,
> *Physical Review Research* **7**, 013044 (2025).
> DOI: [10.1103/PhysRevResearch.7.013044](https://doi.org/10.1103/PhysRevResearch.7.013044)

> Willem Vanderlinden, Enrico Skoruppa, Pauline J. Kolbeck, Enrico Carlon, Jan Lipfert,
> **DNA fluctuations reveal the size and dynamics of topological domains**,
> *PNAS Nexus* **1**(5), pgac268 (2022).
> DOI: [10.1093/pnasnexus/pgac268](https://doi.org/10.1093/pnasnexus/pgac268)

## Contributing

We welcome contributions from the community! To contribute:

- Fork the repository.
- Create a new branch (`git checkout -b feature-branch`).
- Make your changes and commit them (`git commit -am 'Add new feature'`).
- Push to the branch (`git push origin feature-branch`).
- Create a new Pull Request.

Please ensure your code adheres to our coding standards and includes relevant tests.


### Integrated Monte Carlo engine

MDNA includes a fully integrated Monte Carlo simulation engine (`mdna.simulate`) based on [PMCpy](https://github.com/eskoruppa/PMCpy) by Enrico Skoruppa. The original PMCpy code — including its SO3, pyConDec, and PyLk submodules — has been vendored and adapted so that no external git submodules or separate installation steps are required. This engine powers `dna.minimize()` and provides sequence-dependent conformational sampling with features such as excluded volume, automated equilibration, and writhe/linking-number evaluation. See [`mdna/simulate/README.md`](mdna/simulate/README.md) for details on the integration.


### Optional full filament dataset (tutorial)

The filament tutorial runs in minimal mode by default using bundled `filament_minimal` data.
If you want full-trajectory mode, download the optional dataset:

```bash
cd examples
export MDNA_FILAMENT_DATASET_URL='https://doi.org/10.6084/m9.figshare.31423193'
python ./scripts/fetch_filament_dataset.py --output-root ./data
```

Optional integrity check (only if you know the real archive hash):

```bash
cd examples
export MDNA_FILAMENT_DATASET_URL='https://doi.org/10.6084/m9.figshare.31423193'
export MDNA_FILAMENT_DATASET_SHA256='<REAL_SHA256>'
python ./scripts/fetch_filament_dataset.py --output-root ./data --force
```

If you do not have a published SHA256 yet, leave `MDNA_FILAMENT_DATASET_SHA256` unset and run the download command without checksum verification.

What happens behind the scenes: the DOI points to a Figshare item page, and the fetch script automatically resolves the direct file download URL before downloading and unpacking the archive into `examples/data/filament_dataset`.

### Optional: build PyLk Cython extensions (faster + no fallback warnings)

MDNA bundles `PyLk` at `mdna/simulate/Evals/PyLk`. If you see warnings like
"Cython version of writhemap/linkingnumber not compiled", you can compile the extensions in-place:

Prerequisite: ensure `Cython` is installed in the same environment where you run MDNA.

```bash
pip install Cython
```

```bash
# from repository root
cd mdna/simulate/Evals/PyLk
python setup.py build_ext --inplace
```

Quick verification:

```python
from mdna.simulate.Evals.PyLk.pylk import writhemap, linkingnumber
print('PyLk import OK:', writhemap.__name__, linkingnumber.__name__)
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
