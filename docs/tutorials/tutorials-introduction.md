# Tutorial: Generating and Analyzing DNA Structures with mdna

This tutorial will guide you through the process of generating DNA structures, analyzing them, and performing various modifications using the mdna module. We'll cover the following key aspects:

    - Loading DNA Structures
    - Generating DNA Sequences
    - Analyzing DNA Structures
    - Modifying DNA Structures
    - Visualizing DNA

## Loading DNA Structures

The mdna module allows you to load DNA structures from various sources, such as a trajectory file or directly from a sequence.

Example: Loading from a Trajectory
```python
import pymdna as mdna
import mdtraj as md

# Load a trajectory file
traj = md.load('/path_to_your/dna.pdb')

# Load the DNA structure using the trajectory
dna = mdna.load(traj=traj, chainids=[0, 1])
```

Example: Make the DNA structure directly from a sequence
```python
sequence = "ATGCATGC"
dna = mdna.load(sequence=sequence)
```




