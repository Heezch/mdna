# Modifying DNA Structures

MDNA supports three types of modifications on existing DNA structures: **mutation**, **methylation**, and **Hoogsteen flipping**. All modifications update the `Nucleic` object in-place.

---

## Mutations

Replace nucleobases at specific positions using a dictionary of `{index: new_base}`:

```python
import mdna

dna = mdna.make(sequence='AGCGATATAGA')

# Mutate position 0 → G, position 10 → C
dna.mutate({0: 'G', dna.n_bp - 1: 'C'})

dna.describe()
```

### Complementary Strand

By default, the complementary strand is updated automatically (`complementary=True`). To mutate only the leading strand:

```python
dna.mutate({0: 'G'}, complementary=False)
```

### Supported Bases

MDNA supports canonical and non-canonical nucleobases:

| Code | Name | Pairs With | Category |
|------|------|-----------|----------|
| A | Adenine | T | Canonical |
| T | Thymine | A | Canonical |
| G | Guanine | C | Canonical |
| C | Cytosine | G | Canonical |
| U | Uracil | A | RNA incorporation |
| E | 2-Aminopurine (2AP) | T | Fluorescent |
| D | tC (tricyclic cytosine) | G | Fluorescent |
| L | d5SICS | M | Hydrophobic UBP |
| M | dNaM | L | Hydrophobic UBP |
| B | A-analogue | S | Hachimoji |
| S | T-analogue | B | Hachimoji |
| Z | G-analogue | P | Hachimoji |
| P | C-analogue | Z | Hachimoji |

For more details on non-canonical bases, see [Non-Canonical Bases](noncanonical.md).

!!! note
    Mutation clears cached frames, rigid parameters, and minimizer state. Call `get_frames()` or `get_traj()` after mutating to regenerate them.

---

## Methylation

Add methyl groups to cytosine (C5 position) or guanine (O6 position):

```python
dna = mdna.make(sequence='GCGCGCGAGCGA')

# Methylate specific residue indices
dna.methylate(methylations=[0, 2, 4])
```

### CpG Methylation

Automatically methylate all cytosines in CpG context:

```python
dna.methylate(CpG=True, leading_strand=0)
```

!!! warning
    Only C and G residues can be methylated. Attempting to methylate A or T will be silently skipped with a warning message.

---

## Hoogsteen Flipping

Rotate nucleobases around the glycosidic bond to switch between Watson-Crick and Hoogsteen base pairing:

```python
dna = mdna.make(sequence='GCAAAGC')

# Flip base pairs 3 and 4 by 180 degrees (Hoogsteen)
dna.flip(fliplist=[3, 4], deg=180)
```

The `deg` parameter controls the rotation angle. A 180° flip produces the canonical Hoogsteen configuration.

---

## Chaining Modifications

Modifications can be applied sequentially:

```python
dna = mdna.make(sequence='GCGCAATTGCGC')

# 1. Mutate
dna.mutate({0: 'A', 11: 'T'})

# 2. Methylate CpG sites
dna.methylate(CpG=True, leading_strand=0)

# 3. Flip a base pair
dna.flip(fliplist=[5], deg=180)

# 4. Export the modified structure
dna.save_pdb('modified_dna.pdb')
```

---

## Summary

| Modification | Method | Key Arguments |
|-------------|--------|---------------|
| Base substitution | `nucleic.mutate()` | `mutations` dict, `complementary` |
| Methylation | `nucleic.methylate()` | `methylations` list or `CpG=True` |
| Hoogsteen flip | `nucleic.flip()` | `fliplist`, `deg` |
