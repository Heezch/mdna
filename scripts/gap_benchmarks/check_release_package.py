"""Build sdist and wheel offline; exercise the wheel outside the source tree."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SMOKE = r"""
import importlib.metadata
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
import mdna
from mdna.chemistry import COMPONENT_REGISTRY, NUCLEOTIDE_REGISTRY, template_relative_path
from mdna.structure_qc import measure_open_chain_geometry
assert Path(mdna.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())
assert mdna.__version__ == importlib.metadata.version("mdna")
base = Path(mdna.__file__).parent
for code in NUCLEOTIDE_REGISTRY:
    assert (base / template_relative_path(code)).is_file(), code
model = mdna.make(sequence="ACGTACGT").get_traj()
qc = measure_open_chain_geometry(model)
assert model.n_residues == 16 and len(qc["links"]) == 14 and qc["missing_links"] == 0
for name in ("build_rna_chain_from_frames", "close_rna_gap_from_anchors", "close_dna_gap_from_anchors"):
    assert callable(getattr(mdna, name))
print("MDNA_SMOKE=" + json.dumps(dict(version=mdna.__version__, residues=model.n_residues,
      expected_links=len(qc["links"]), missing_links=qc["missing_links"],
      registered_dna_templates=len(NUCLEOTIDE_REGISTRY),
      rna_component_json=(base / "data/rna_components.json").is_file())))
"""


def check():
    with TemporaryDirectory(prefix="mdna-release-") as tmp:
        build_root = Path(tmp)
        source = build_root / "source"
        source.mkdir()
        for file in ("pyproject.toml", "MANIFEST.in", "README.md", "LICENSE", "setup.py"):
            shutil.copy2(ROOT / file, source / file)
        shutil.copytree(ROOT / "mdna", source / "mdna", ignore=shutil.ignore_patterns("__pycache__", "build"))
        dist = build_root / "dist"
        env = dict(os.environ, PIP_NO_INDEX="1", PYTHONPATH="", PYTHONDONTWRITEBYTECODE="1")
        subprocess.run([sys.executable, "-B", "-m", "build", "--no-isolation", "--outdir", str(dist)],
                       cwd=source, env=env, check=True)
        wheel, = dist.glob("*.whl")
        sdist, = dist.glob("*.tar.gz")
        payload = build_root / "payload"
        with zipfile.ZipFile(wheel) as archive:
            members = archive.namelist()
            if len(members) != len(set(members)):
                raise ValueError("wheel contains duplicate members")
            if any(n.endswith((".so", ".pyd", "_cython.c", "_cython.html")) for n in members):
                raise ValueError("wheel includes local compiled artifacts")
            archive.extractall(payload)
        smoke = subprocess.run([sys.executable, "-I", "-B", "-c", SMOKE, str(payload)],
                               cwd=build_root, env=env, capture_output=True, text=True)
        if smoke.returncode:
            raise RuntimeError(smoke.stdout + smoke.stderr)
        observed = [line.removeprefix("MDNA_SMOKE=") for line in smoke.stdout.splitlines()
                    if line.startswith("MDNA_SMOKE=")]
        result, = observed
        checks = ("test_generator_contract.py", "test_geometry_contract.py", "test_rna_gap_closure.py",
                  "test_duplex_mmcif.py")
        for name in checks:
            shutil.copy2(ROOT / "tests" / name, build_root / name)
        fixture_root = ROOT / "artifacts/rna_pilot"
        fixture_copy = build_root / "artifacts/rna_pilot"
        ion_copy = fixture_copy / "reference/ion_series"
        ion_copy.mkdir(parents=True)
        shutil.copy2(fixture_root / "ideal_A_dsRNA_UCGUACGA.pdb", fixture_copy)
        ion_root = fixture_root / "reference/ion_series"
        for file in (*ion_root.glob("*.cif"), ion_root / "x3dna_published_parameters.json"):
            shutil.copy2(file, ion_copy)
        runner = ("import sys; sys.path.insert(0, sys.argv.pop(1)); "
                  "import mdna, pathlib; "
                  "assert pathlib.Path(mdna.__file__).resolve().is_relative_to(pathlib.Path(sys.path[0])); "
                  "import pytest; raise SystemExit(pytest.main(sys.argv[1:]))")
        regression = subprocess.run(
            [sys.executable, "-I", "-B", "-c", runner, str(payload), "-q",
             "-p", "no:cacheprovider", "--import-mode=importlib", *checks],
            cwd=build_root, env=env, capture_output=True, text=True)
        if regression.returncode:
            raise RuntimeError(regression.stdout + regression.stderr)
        return dict(wheel=wheel.name, wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
                    sdist=sdist.name, sdist_sha256=hashlib.sha256(sdist.read_bytes()).hexdigest(),
                    wheel_files=len(members), smoke=json.loads(result),
                    wheel_regression_files=checks, wheel_regression_output=regression.stdout,
                    scope="Unpacked wheel outside source tree with the existing dependency environment")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(check(), indent=2, allow_nan=False)+"\n", encoding="utf-8")
