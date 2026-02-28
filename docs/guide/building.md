# Building DNA Structures

This guide covers all the ways to create DNA structures with MDNA: from simple sequences to custom-shaped assemblies.

## Basic Construction

### From a Sequence

```python
import mdna

dna = mdna.make(sequence='GCGCGCGCGC')
dna.describe()
dna.draw()
```

### From a Number of Base Pairs

When no sequence is provided, a random sequence is generated:

```python
dna = mdna.make(n_bp=50)
print(dna.sequence)
```

### From Existing Data

Load from an MDTraj trajectory or PDB file:

```python
import mdtraj as md

# From a PDB file
dna = mdna.load(filename='my_dna.pdb')

# From an MDTraj trajectory
traj = md.load('trajectory.xtc', top='topology.pdb')
dna = mdna.load(traj=traj, chainids=[0, 1])
```

Or from precomputed reference frames:

```python
import numpy as np
frames = np.load('frames.npy')  # shape (n_bp, 4, 3)
dna = mdna.load(frames=frames, sequence='ATCG...')
```

---

## Custom Shapes

MDNA uses spline interpolation through control points to define arbitrary DNA paths.

### Built-in Shapes

The `Shapes` class provides common geometries:

```python
# Straight line (default)
points = mdna.Shapes.line(length=5)

# Circle
points = mdna.Shapes.circle(radius=3)

# Helix / supercoil
points = mdna.Shapes.helix(height=3, pitch=5, radius=7, num_turns=4)
```

Use them with `make()`:

```python
dna = mdna.make(n_bp=300, control_points=mdna.Shapes.helix())
dna.draw()
```

### Custom Control Points

Define any 3D path with at least 4 points:

```python
import numpy as np

control_points = np.array([
    [0, 0, 0],
    [30, 10, -10],
    [50, 10, 20],
    [3, 4, 30]
])

dna = mdna.make(n_bp=100, control_points=control_points)
dna.draw()
```

!!! tip
    When only `control_points` are provided (no `sequence` or `n_bp`), MDNA infers `n_bp` from the spline arc length using the standard 0.34 nm rise per base pair.

---

## Circular DNA

Create closed minicircles with optional linking number control:

```python
# Relaxed minicircle
dna = mdna.make(n_bp=200, circular=True)
print('Lk, Wr, Tw:', dna.get_linking_number())
dna.draw()

# Supercoiled minicircle (overwound by 8 turns)
dna = mdna.make(n_bp=200, circular=True, dLk=8)
print('Lk, Wr, Tw:', dna.get_linking_number())
```

!!! note
    When minimizing supercoiled DNA, use `equilibrate_writhe=True` to allow the writhe to equilibrate:
    ```python
    dna.minimize(equilibrate_writhe=True)
    ```

---

## Minimization

Monte Carlo relaxation resolves steric clashes and elastic strain:

```python
dna = mdna.make(n_bp=100)
dna.minimize(
    temperature=300,      # Kelvin
    exvol_rad=2.0,        # Excluded volume radius
    endpoints_fixed=True  # Pin the ends
)
```

You can also fix specific base pairs:

```python
dna.minimize(fixed=[0, 1, 2, 97, 98, 99])
```

To view the Monte Carlo trajectory:

```python
mc_traj = dna.get_MC_traj()
```

---

## Extending DNA

Add base pairs to the 3' or 5' end of an existing structure:

```python
dna = mdna.make(n_bp=50)

# Extend forward (3' direction)
dna.extend(sequence='G' * 30, forward=True)

# Extend backward (5' direction)
dna.extend(sequence='C' * 20, forward=False)

dna.draw()
```

---

## Connecting Two DNA Structures

Join two `Nucleic` objects with an automatically optimized linker:

```python
import numpy as np

dna0 = mdna.make(sequence='AAAAAAAAA')
dna1 = mdna.make(
    sequence='GGGGGGGGG',
    control_points=mdna.Shapes.line(1) + np.array([4, 0, -5])
)

# connect() finds the optimal number of linker base pairs
connected = mdna.connect(dna0, dna1)
connected.describe()
connected.draw()
```

The `connect()` function:

1. Computes the twist difference between the two end frames
2. Finds the optimal number of linking base pairs to achieve neutral twist
3. Interpolates a connecting path between the two structures
4. Optionally minimizes the resulting assembly

---

## Exporting Structures

### Save as PDB

```python
dna.save_pdb('output.pdb')
```

### Get MDTraj Trajectory

```python
traj = dna.get_traj()
traj.save('output.h5')
```

### Direct PDB Generation

For one-shot PDB generation without creating a `Nucleic` object:

```python
traj = mdna.sequence_to_pdb(
    sequence='CGCGAATTCGCG',
    filename='my_dna',
    output='GROMACS'
)
```

---

## Summary

| Task | Function / Method |
|------|-------------------|
| Generate from sequence/shape | `mdna.make()` |
| Load from trajectory/frames | `mdna.load()` |
| Relax structure | `nucleic.minimize()` |
| Extend DNA | `nucleic.extend()` |
| Join two structures | `mdna.connect()` |
| Export PDB | `nucleic.save_pdb()` |
| One-shot PDB | `mdna.sequence_to_pdb()` |
| MD simulation | `mdna.sequence_to_md()` |
