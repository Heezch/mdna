# MDNA: Molecular DNA Structure Generation and Analysis Toolkit

## Description
MDNA is a powerful molecular modeling tool designed to bridge the gap in generating starting structures for DNA and DNA-protein complexes for molecular dynamics (MD) simulations. By leveraging rigid body models, this tool enhances the precision in representing DNA structures, allowing for the inclusion of non-canonical bases, the adjustment of Watson-Crick to Hoogsteen configurations, methylation patterns, and structural extensions. It also provides robust analytical capabilities for MD simulations, calculating geometric properties, curvature, and protein-DNA interaction profiles.

## Installation
To install MDNA use pip:
```bash
pip install mdna
```

Or if you want to install the most recent version of MDNA, follow these steps:
```bash
git clone --recurse-submodules https://github.com/heezch/mdna.git
```
After that go to the 'mdna' project folder and do:
```bash
pip install .
```


## Usage

See our [documentation page](https://heezch.github.io/mdna/)

## Contributing

We welcome contributions from the community! To contribute:

- Fork the repository.
- Create a new branch (`git checkout -b feature-branch`).
- Make your changes and commit them (`git commit -am 'Add new feature'`).
- Push to the branch (`git push origin feature-branch`).
- Create a new Pull Request.

Please ensure your code adheres to our coding standards and includes relevant tests.

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

MDNA vendors `PyLk` inside `mdna/PMCpy/pmcpy/Evals/PyLk`. If you see warnings like
"Cython version of writhemap/linkingnumber not compiled", you can compile the extensions in-place:

Prerequisite: ensure `Cython` is installed in the same environment where you run MDNA.

```bash
# example for conda/mamba env named `mdna`
mamba run -n mdna python -m pip install Cython
```

```bash
# from repository root
cd mdna/PMCpy/pmcpy/Evals/PyLk
python setup.py build_ext --inplace
```

If you use conda/mamba, run with your env explicitly:

```bash
mamba run -n mdna python mdna/PMCpy/pmcpy/Evals/PyLk/setup.py build_ext --inplace
```

Quick verification:

```bash
mamba run -n mdna python - <<'PY'
from mdna.PMCpy.pmcpy.Evals.PyLk.pylk import writhemap, eval_link
print('PyLk import OK:', writhemap.__name__, eval_link.__name__)
PY
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
