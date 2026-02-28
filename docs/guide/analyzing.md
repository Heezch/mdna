# Analyzing DNA Structures

MDNA provides tools for computing rigid base parameters, linking numbers, and groove widths from DNA structures and MD trajectories.

---

## Loading a Trajectory

```python
import mdna

# From files
dna = mdna.load(filename='trajectory.xtc', top='topology.pdb')

# Or from an existing MDTraj trajectory
import mdtraj as md
traj = md.load('trajectory.xtc', top='topology.pdb')
dna = mdna.load(traj=traj, chainids=[0, 1])

dna.describe()
```

---

## Rigid Base Parameters

The rigid base formalism describes DNA geometry through 12 parameters split into two groups:

### Base Pair Parameters (intra-base pair)

Describe the relative position/orientation of two bases within a pair:

| Parameter | Type | Unit |
|-----------|------|------|
| Shear | Translation | nm |
| Stretch | Translation | nm |
| Stagger | Translation | nm |
| Buckle | Rotation | degrees |
| Propeller | Rotation | degrees |
| Opening | Rotation | degrees |

### Base Pair Step Parameters (inter-base pair)

Describe the relative position/orientation between consecutive base pair steps:

| Parameter | Type | Unit |
|-----------|------|------|
| Shift | Translation | nm |
| Slide | Translation | nm |
| Rise | Translation | nm |
| Tilt | Rotation | degrees |
| Roll | Rotation | degrees |
| Twist | Rotation | degrees |

### Computing Parameters

```python
# Get all 12 parameters
params, names = dna.get_parameters()
print(names)   # List of parameter names
print(params.shape)  # (n_frames, n_bp, 12)

# Get only step parameters
step_params, step_names = dna.get_parameters(step=True)

# Get only base pair parameters
bp_params, bp_names = dna.get_parameters(base=True)

# Get a single parameter by name
twist = dna.get_parameter('twist')
roll = dna.get_parameter('roll')
```

### Plotting Parameters

The `NucleicFrames` object provides built-in plotting:

```python
rigid = dna.get_rigid_object()
rigid.plot_parameters()
```

### Custom Visualization

```python
import matplotlib.pyplot as plt

params, names = dna.get_parameters(step=True)

fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for i, (ax, name) in enumerate(zip(axes.flat, names)):
    data = params[:, :, i]
    ax.plot(data.mean(axis=0))
    ax.fill_between(
        range(data.shape[1]),
        data.mean(axis=0) - data.std(axis=0),
        data.mean(axis=0) + data.std(axis=0),
        alpha=0.3
    )
    ax.set_title(name)
plt.tight_layout()
```

---

## Rigid Base Parameters from a Trajectory

For standalone analysis without creating a `Nucleic` object:

```python
import mdtraj as md

traj = md.load('dna.pdb')
rigid = mdna.compute_rigid_parameters(traj, chainids=[0, 1])
```

This returns a `NucleicFrames` object with the same interface as `dna.get_rigid_object()`.

---

## Base Reference Frames

Access the individual base reference frames (useful for custom geometry analysis):

```python
base_frames = dna.get_base_frames()
# Returns dict: {residue_topology: frames (n_frames, 4, 3)}
```

The frame rows are: origin, $\hat{b}_L$ (long axis), $\hat{b}_D$ (short axis), $\hat{b}_N$ (normal).

---

## Linking Number

For circular DNA, compute the linking number and its decomposition into writhe and twist:

```python
dna = mdna.make(n_bp=200, circular=True, dLk=2)
Lk, Wr, Tw = dna.get_linking_number()
print(f'Lk = {Lk:.2f}, Wr = {Wr:.2f}, Tw = {Tw:.2f}')
```

The linking number is computed using Gauss's linking integral via the PyLk library.

---

## Groove Width Analysis

The `GrooveAnalysis` class computes major and minor groove widths by fitting cubic splines through phosphorus atom positions:

```python
import mdtraj as md

traj = md.load('dna_trajectory.xtc', top='dna.pdb')
grooves = mdna.GrooveAnalysis(traj)

# Plot groove widths
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 3))
grooves.plot_groove_widths(ax=ax)
```

---

## Torsion Angle Analysis

The `TorsionAnalysis` class computes backbone torsion angles:

```python
torsions = mdna.TorsionAnalysis(traj)
```

---

## Summary

| Analysis | Function / Method | Returns |
|----------|-------------------|---------|
| All rigid base parameters | `nucleic.get_parameters()` | `(params, names)` |
| Single parameter | `nucleic.get_parameter('twist')` | `np.ndarray` |
| NucleicFrames object | `nucleic.get_rigid_object()` | `NucleicFrames` |
| Standalone rigid params | `mdna.compute_rigid_parameters(traj)` | `NucleicFrames` |
| Base frames | `nucleic.get_base_frames()` | `dict` |
| Linking number | `nucleic.get_linking_number()` | `[Lk, Wr, Tw]` |
| Groove widths | `mdna.GrooveAnalysis(traj)` | `GrooveAnalysis` |
| Torsion angles | `mdna.TorsionAnalysis(traj)` | `TorsionAnalysis` |
