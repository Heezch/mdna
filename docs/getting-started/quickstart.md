# Quickstart

This page gets you from zero to a generated, minimized, and analyzed DNA structure in under five minutes.

## 1. Generate a DNA Structure

```python
import mdna

# From a sequence
dna = mdna.make(sequence='ATCGATCGGT')
dna.describe()
```

MDNA returns a `Nucleic` object — the central data structure that holds the sequence, reference frames, and (optionally) an atomic-resolution MDTraj trajectory.

## 2. Visualize

```python
dna.draw()
```

This produces a 3D matplotlib plot showing the helical axis and backbone.

## 3. Minimize the Structure

Monte Carlo relaxation removes steric clashes and elastic strain:

```python
dna.minimize()
dna.draw()
```

!!! note
    `minimize()` updates the internal trajectory and frames in-place.

## 4. Analyze Rigid Base Parameters

```python
params, names = dna.get_parameters()
print("Parameter names:", names)
print("Shape:", params.shape)  # (n_frames, n_bp, n_params)
```

Or retrieve a single parameter:

```python
twist = dna.get_parameter('twist')
```

## 5. Export to PDB

```python
dna.save_pdb('my_dna.pdb')
```

Or get the MDTraj trajectory directly:

```python
traj = dna.get_traj()
```

---

## What's Next?

| Goal | Page |
|------|------|
| Build custom shapes, circular DNA, extend & connect | [Building DNA](../guide/building.md) |
| Mutate, methylate, flip bases | [Modifying DNA](../guide/modifying.md) |
| Analyze trajectories and rigid base parameters | [Analyzing DNA](../guide/analyzing.md) |
| Understand how the classes fit together | [Architecture](../concepts/architecture.md) |
| Full function/class reference | [API Reference](../api/index.md) |
