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

MDNA was developed to address a need in molecular dynamics (MD) simulations, particularly for generating and analyzing DNA structures and DNA-protein complexes. Unlike proteins, which have abundant experimental data for structure generation, DNA often requires construction from scratch, especially when dealing with non-canonical bases or specific structural motifs like Hoogsteen base pairs. MDNA bridges this gap by providing a versatile and accurate tool that simplifies the generation of complex DNA structures and improves the precision of the generation of starting structures for MD simulations.


The development of MDNA was driven by the limitations of existing tools that either lacked flexibility or required specialized knowledge to use effectively. These tools often struggled with complex DNA configurations or integrating into broader research workflows. MDNA was designed as an all-in-one solution that consolidates key functionalities into a cohesive Python ecosystem (MDTraj, OpenMM, OpenPathSampling, etc.), making DNA structure generation and analysis more accessible and efficient.

## Working with MDNA

MDNA offers robust capabilities for generating and analyzing DNA structures. Researchers can create DNA models by specifying sequences, topologies, or custom shapes, and manipulate configurations to study specific structural motifs. MDNA enables researchers to explore the dynamic behavior of DNA and its interactions by providing tools to analyze rigid base parameters, linking numbers, and more. This comprehensive approach helps bridge the gap between DNA's static structure and its dynamic functions, making MDNA an invaluable resource for structural biology, genomics, and beyond.
