# Non-Canonical Nucleobases

MDNA supports a wide range of non-canonical and synthetic nucleobases beyond the standard A, T, G, C alphabet. This page describes all supported bases and how to use them.

---

## Complementary Base Pairing Map

```python
complementary_map = {
    'A': 'T',  'T': 'A',  'G': 'C',  'C': 'G',   # Canonical
    'U': 'A',                                        # RNA
    'E': 'T',  'D': 'G',                            # Fluorescent
    'L': 'M',  'M': 'L',                            # Hydrophobic UBP
    'B': 'S',  'S': 'B',  'Z': 'P',  'P': 'Z'      # Hachimoji
}
```

---

## Base Categories

### Canonical Bases

| Code | Name | Pairs With |
|------|------|-----------|
| A | Adenine | T |
| T | Thymine | A |
| G | Guanine | C |
| C | Cytosine | G |

### RNA Incorporation

| Code | Name | Pairs With | Reference |
|------|------|-----------|-----------|
| U | Uracil | A | — |

Represents RNA incorporation into a DNA duplex.

### Fluorescent Bases

| Code | Name | Pairs With | Reference |
|------|------|-----------|-----------|
| E | 2-Aminopurine (2AP) | T | Ward et al., 1969 |
| D | tC (tricyclic cytosine) | G | Wilhelmsson et al., 2003 |

2-Aminopurine is widely used as a fluorescent probe due to its sensitivity to the local environment. The tricyclic cytosine analogue (tC) is known for its unique photophysical properties and high quantum yield.

### Hydrophobic Unnatural Base Pairs

| Code | Name | Pairs With | Reference |
|------|------|-----------|-----------|
| L | d5SICS | M | Malyshev et al., 2014 |
| M | dNaM | L | Malyshev et al., 2014 |

These hydrophobic pairs maintain duplex stability without hydrogen bonding, demonstrating that shape complementarity alone can support base pairing.

### Hachimoji Bases

| Code | Name | Pairs With | Reference |
|------|------|-----------|-----------|
| B | A-analogue (isoG) | S | Hoshika et al., 2019 |
| S | T-analogue (isoC) | B | Hoshika et al., 2019 |
| Z | G-analogue | P | Hoshika et al., 2019 |
| P | C-analogue | Z | Hoshika et al., 2019 |

The Hachimoji ("eight-letter") DNA system extends the genetic alphabet from 4 to 8 bases, forming two new orthogonal pairs (B–S and P–Z).

---

## Usage

### Constructing with Non-Canonical Bases

You can include non-canonical bases directly in the sequence:

```python
import mdna

# DNA with hachimoji bases
dna = mdna.make(sequence='ATBSCGPZ')
dna.describe()
```

### Mutating to Non-Canonical Bases

```python
dna = mdna.make(sequence='AGCGATATAGA')

# Introduce fluorescent base at position 0 and hydrophobic pair at position 5
dna.mutate({0: 'E', 5: 'L'}, complementary=True)
dna.describe()
```

### Inspecting the Complementary Map

```python
# The Nucleic object carries the pairing map
print(dna.base_pair_map)
```

---

## Structural Notes

All non-canonical bases use reference geometries stored as HDF5 files in `mdna/atomic/bases/`. The reference frame convention follows the Tsukuba convention, with the glycosidic bond atom varying by base type:

- **Purines** (A, G, E, B, P): N9–C4 convention
- **Pyrimidines** (C, T, U, D): N1–C2 convention
- **Hachimoji pyrimidines** (S, Z): C1–C2 convention
- **Hydrophobic** (L): N1–C5 convention
- **Hydrophobic** (M): C1–C6 convention

These conventions ensure correct frame placement and base pair geometry for all supported nucleotides.
