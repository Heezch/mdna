<!-- This part of the project documentation focuses on
an **information-oriented** approach. Use it as a
reference for the technical implementation of the
`mdna` project code.

 -->
This section of the documentation provides detailed, information-oriented references for the technical implementation of the `mdna` project code. It is intended as a comprehensive guide for developers and users who wish to understand the inner workings of the MDNA tool, its modules, and functions.

## Nucleic class

The `pymdna.nucleic` module is the core of the MDNA toolkit, encompassing a variety of classes and functions essential for DNA structure generation, manipulation, and analysis. Below, each key component of the module is outlined with explanations of its purpose and usage. The `Nucleic` class serves as the primary interface for interacting with DNA structures in the MDNA toolkit. It encapsulates both the structural properties of DNA and the trajectory information needed for molecular dynamics simulations. Key methods include:


::: pymdna.nucleic

<!-- ### Nucleic Class

The `Nucleic` class serves as the primary interface for interacting with DNA structures in the MDNA toolkit. It encapsulates both the structural properties of DNA and the trajectory information needed for molecular dynamics simulations. Key methods include:

- **load()**: Initializes a DNA structure from either a base step reference frame or an MDTraj trajectory. This function ensures that the DNA structure is correctly represented and returns a `Nucleic` object.
- **make()**: Generates a DNA structure from scratch, allowing users to specify sequences, topologies (e.g., circular or linear), and shapes via control points.
- **minimize()**: Performs energy minimization using Monte Carlo simulations to relax and optimize the generated DNA structure.
- **mutate()**: Alters the nucleotide sequence, supporting both canonical and non-canonical bases, including synthetic and fluorescent bases.
- **methylate()**: Adds methylation patterns to cytosines, particularly at CpG sites, reflecting common epigenetic modifications.
- **flip()**: Rotates nucleobases around the glycosidic bond, converting between Watson-Crick and Hoogsteen base-pairing configurations.
- **extend()**: Adds additional base pairs to an existing DNA structure, extending the sequence in the 5' or 3' direction with options for custom shapes.
- **connect()**: Links two separate DNA structures by generating a new strand that joins them, with customizable parameters for sequence and topology.

### Analysis Functions

The `pymdna.nucleic` module also includes a suite of functions designed for the analysis of DNA structures and trajectories:

- **compute_rigid_parameters()**: Calculates rigid base parameters such as translation and rotation between base pairs, essential for understanding DNA's structural dynamics.
- **compute_linking_number()**: Determines the linking number, a topological property crucial for understanding DNA supercoiling and its biological implications.
- **compute_curvature()**: Analyzes DNA bending, a key factor in processes like gene regulation and DNA packaging.
- **compute_groove_widths()**: Measures the major and minor groove widths of DNA, which are important for studying DNA-protein interactions.

### Integration with MDTraj

MDNA is designed to integrate seamlessly with the MDTraj library, which facilitates the handling of MD data formats and computations. The `Nucleic` class methods support the direct manipulation of MDTraj objects, allowing for easy retrieval of updated configurations at atomic resolution. This integration enables users to transition from structure generation to MD simulation efficiently.

### Example Usage

```python
import pymdna.nucleic as mdna

# Load a DNA structure from a trajectory
dna = mdna.load("trajectory.xtc")

# Generate a new DNA structure
new_dna = mdna.make(sequence="ATCG", circular=True)

# Minimize the energy of the structure
minimized_dna = new_dna.minimize()

# Analyze rigid base parameters
rigid_params = dna.compute_rigid_parameters()

# Extend the DNA structure
extended_dna = dna.extend(n_bp=50, forward=True)

# Connect two DNA structures
connected_dna = mdna.connect(dna1, dna2) -->
