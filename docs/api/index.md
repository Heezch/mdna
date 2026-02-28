# API Reference

Complete reference for all public classes and functions in the `mdna` package.

---

## Top-Level Functions

These are available directly via `import mdna`:

| Function | Description |
|----------|-------------|
| [`make()`](functions.md#mdna.nucleic.make) | Generate DNA from sequence and/or shape |
| [`load()`](functions.md#mdna.nucleic.load) | Load DNA from trajectory, frames, or sequence |
| [`connect()`](functions.md#mdna.nucleic.connect) | Join two DNA structures |
| [`compute_rigid_parameters()`](functions.md#mdna.nucleic.compute_rigid_parameters) | Compute rigid base parameters from a trajectory |
| [`sequence_to_pdb()`](functions.md#mdna.nucleic.sequence_to_pdb) | Generate PDB file directly from sequence |
| [`sequence_to_md()`](functions.md#mdna.nucleic.sequence_to_md) | Generate and simulate DNA with OpenMM |

## Core Classes

| Class | Description |
|-------|-------------|
| [`Nucleic`](nucleic.md) | Central DNA structure object (sequence + frames + trajectory) |
| [`Shapes`](shapes.md) | Factory for predefined DNA shape control points |

## Geometry & Analysis

| Class | Description |
|-------|-------------|
| [`NucleicFrames`](geometry.md#rigid-base-parameter-class) | Rigid base parameter computation from trajectories |
| [`ReferenceBase`](geometry.md#reference-base-class) | Single nucleobase reference frame fitting |
| [`GrooveAnalysis`](analysis.md#groove-analysis) | DNA groove width computation |
| [`TorsionAnalysis`](analysis.md#torsion-analysis) | Backbone torsion angle computation |

## Build Internals

| Class | Description |
|-------|-------------|
| [`SplineFrames`](spline.md) | Spline interpolation and frame generation |
| [`SequenceGenerator`](generators.md#sequence-generator) | DNA topology construction from sequence |
| [`StructureGenerator`](generators.md#structure-generator) | Atomic coordinate placement into frames |

## Relaxation

| Class | Description |
|-------|-------------|
| [`Minimizer`](minimizer.md) | Monte Carlo energy minimization |
