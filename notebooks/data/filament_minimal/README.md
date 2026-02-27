# H-NS Filament Minimal Dataset

This folder contains a compact, package-friendly subset of the full filament MD dataset.

## Purpose

The goal is to avoid shipping the full ~1.7 GB trajectory bundle while preserving deterministic filament construction behavior for the tutorial's non-random assembly path.

This minimal dataset is designed for:

- `Assembler.load_minimal_site_map(...)`
- `assembler.add_dimer(segment='minimal')`

## Contents

- `s1s1_start.pdb`
- `s2s2_start.pdb`
- `s1s1_extend.pdb`
- `s2s2_extend.pdb`
- `complex_frame_1.pdb`
- `manifest.json`

## Size

- Total: ~`1.5M`

## How it works

Unlike the full dataset (many `.xtc` trajectories), this folder stores only a few selected **source-state PDB frames**:

- start-state source frames for `s1s1` and `s2s2`
- extend-state source frames for `s1s1` and `s2s2`
- one DBD-DNA complex frame

At runtime, `Assembler.load_minimal_site_map(...)`:

1. Loads these source PDBs
2. Rebuilds segment/site maps using `SiteMapper`
3. Selects the required site structures for start/extend assembly
4. Uses `complex_frame_1.pdb` for `add_dna(...)`

So the segmentation logic is still performed in code (like full mode), but from minimal source inputs.

## Generation

This dataset is generated from `examples/data/filament_dataset` with:

`python examples/scripts/extract_minimal_filament_dataset.py --output-dir examples/data/filament_minimal`

The selection metadata (trajectory ids, stride, source mapping) is stored in `manifest.json`.

## Notes

- `segment='minimal'` is intended for reproducible compact usage.
- `segment='fixed'` and `segment='random'` remain tied to full trajectory-style site maps.
- If you regenerate with a different `--dna-frame-idx`, the output complex file name and manifest should stay consistent for your workflow.
