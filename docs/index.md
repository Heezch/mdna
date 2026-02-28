# MDNA — DNA Structure Generation & Analysis Toolkit

A Python toolkit for atomic-resolution generation and analysis of double-stranded DNA structures.

---

## What is MDNA?

MDNA enables the construction of arbitrarily shaped DNA using spline-based mapping, supports canonical and non-canonical nucleotides, and integrates Monte Carlo relaxation to obtain physically consistent configurations. In addition, it implements rigid base parameter analysis and linking number calculations, and exports directly to MDTraj-compatible trajectories for molecular dynamics workflows.

## Quick Example

```python
import mdna

# Generate a 100-bp DNA minicircle
dna = mdna.make(n_bp=100, circular=True)

# Relax the structure
dna.minimize()

# Compute rigid base parameters
params, names = dna.get_parameters()

# Export to PDB
dna.save_pdb('minicircle.pdb')
```

## Key Features

- **Arbitrary DNA shapes** via spline control points
- **Sequence-driven construction** with canonical and non-canonical bases (hachimoji, fluorescent, hydrophobic UBPs)
- **Hoogsteen base flipping** and **methylation** editing
- **Circular DNA** generation with linking number control ($\Delta Lk$)
- **Monte Carlo relaxation** for physically consistent configurations
- **Rigid base parameter analysis**: shear, stretch, stagger, buckle, propeller, opening, shift, slide, rise, tilt, roll, twist
- **MDTraj interoperability** for seamless MD workflows

---

## Documentation

## Documentation

| | Section | Description |
|---|---------|-------------|
| :material-rocket-launch: | **[Getting Started](getting-started/installation.md)** | Install MDNA and generate your first DNA in 5 minutes |
| :material-book-open-variant: | **[User Guide](guide/overview.md)** | Task-oriented guides: [Build](guide/building.md) · [Modify](guide/modifying.md) · [Analyse](guide/analyzing.md) |
| :material-lightbulb-on: | **[Concepts](concepts/architecture.md)** | Architecture, splines, rigid base formalism |
| :material-notebook: | **[Jupyter Notebooks](index-notebooks.md)** | Interactive tutorials from basic to advanced |
| :material-code-tags: | **[API Reference](api/index.md)** | Complete reference for all classes and functions |

## Example Gallery

Three examples that highlight the building of biomolecular assemblies with MDNA: extension of DNA structures, using proteins as scaffold to generate DNA structure, and connecting two DNA strands to form a DNA loop. Molecular representations are visualized with Mol* Viewer.

<div class="image-gallery">
  <a href="assets/gallery/image1.png" class="glightbox">
    <img src="assets/gallery/image1.png" alt="DNA Extension" />
  </a>
  <a href="assets/gallery/image2.png" class="glightbox">
    <img src="assets/gallery/image2.png" alt="Protein-Scaffolded DNA" />
  </a>
  <a href="assets/gallery/image3.png" class="glightbox">
    <img src="assets/gallery/image3.png" alt="DNA Loop Connection" />
  </a>
</div>

## Citation

Link to the [publication](https://www.biorxiv.org/content/10.1101/2025.07.26.666940v1.abstract)

## Acknowledgements

This project is supported by the NWO Klein grant.
