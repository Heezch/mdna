<!-- This part of the project documentation focuses on an
**understanding-oriented** approach. You'll get a
chance to read about the background of the project,
as well as reasoning about how it was implemented.

> **Note:** Expand this section by considering the
> following points:

- Give context and background on your library
- Explain why you created it
- Provide multiple examples and approaches of how
    to work with it
- Help the reader make connections
- Avoid writing instructions or technical descriptions
    here -->
## Background and Context

MDNA was created to support molecular dynamics simulations that require precise construction and analysis of double stranded DNA and DNA protein assemblies. In contrast to proteins, where experimentally resolved structures are often available, DNA systems frequently need to be built from first principles, particularly when introducing non canonical nucleotides, methylation patterns, altered linking numbers, or Hoogsteen base pairing. MDNA addresses this challenge by enabling atomic resolution DNA generation from sequence and geometry, producing reliable starting configurations for simulation studies.

Many existing nucleic acid tools focus either on visualization, coarse grained modeling, or rigid body analysis, and often lack flexibility for constructing arbitrarily shaped DNA or integrating seamlessly into simulation pipelines. MDNA was therefore designed as a unified Python framework that combines spline based structure generation, Monte Carlo relaxation, sequence editing, and rigid base analysis within a single ecosystem. It interoperates directly with widely used simulation libraries such as MDTraj and OpenMM, enabling smooth transitions from model building to production simulations and trajectory analysis.


## Working with MDNA

MDNA provides a flexible workflow for constructing and interrogating DNA structures. Users can generate models from sequence alone, define custom geometries through control points, enforce circular topologies with prescribed linking number differences, or extend and connect existing fragments to build large heterogeneous assemblies. Optional Monte Carlo minimization relaxes elastic deformations and steric clashes while preserving topological constraints when required.

For structural analysis, MDNA implements the full rigid base formalism natively in Python. Intra base pair parameters include shear, stretch, stagger, buckle, propeller, and opening. Inter base pair step parameters include shift, slide, rise, tilt, roll, and twist. The toolkit also provides direct computation of linking number, including its decomposition into twist and writhe.

Although MDNA does not currently include dedicated curvature or persistence length functions, the rigid base frames and step parameters it generates provide all required geometric information to compute such quantities externally. Users can derive curvature from base pair frame orientations and estimate persistence length from tangent correlation analysis using the exported trajectory data.
