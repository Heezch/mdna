
# Tutorial: Generating and Analyzing DNA Structures with mdna

This tutorial will guide you through the process of generating DNA structures, analyzing them, and performing various modifications using the mdna module. We'll cover how to modify DNA Structures:

- Mutations
- Methylation
- Hoogsteen flips


### Example: Make the DNA structure directly from a sequence
```python

import mdna as mdna
import mdtraj as md

sequence = "ATGCATGC"
dna = mdna.load(sequence=sequence)
```

## Modifying DNA Structures

The mdna module offers various functions to modify DNA structures, such as mutating bases, methylating sites, and performing Hoogsteen flips. Below are examples of how to perform these modifications.


You can mutate specific bases in the DNA sequence. In the following example, the first base is mutated to a G, and the last base is mutated to a C.

```python
# Here we make a DNA with the following sequence
dna = mdna.make(sequence='AGCGATATAGA')

# Let's save the original structure
traj = dna.get_traj()
traj.save_pdb('dna_original.pdb')

# Modify the DNA sequence
# Mutate the first base to a G and the last base to a C
dna.mutate({0: 'G', dna.n_bp - 1: 'C'}, complementary=False)

# Get information about the DNA and see the mutated sequence
dna.describe()
```


Methylation can be applied to specific positions or all CpG sites. Below are examples of both approaches.

```python
# Methylate the 5th position, which is T (this will fail and is caught by the function)
dna.methylate(methylations=[5])

# Methylate all CpG sites
dna.methylate(CpG=True)
```

You can perform a Hoogsteen flip on any base pair. In the example below, the 5th base pair is flipped by 180 degrees.

```python
# Flip the 5th base pair with a 180-degree rotation
dna.flip(fliplist=[5], deg=180)
```

### Saving the Modified Structure

After making modifications, you can save the modified DNA structure to a file.

```python
# Get trajectory of the modified DNA or save it as a pdb file
traj_mod = dna.get_traj()
traj_mod.save_pdb('dna_modified.pdb')
```

This concludes the tutorial on generating, analyzing, and modifying DNA structures using the mdna module. In the next tutorial, we will explore advanced visualization techniques for DNA structures.



# Modifying DNA Structures with Non-Canonical Nucleobases

The `mdna` module not only allows you to work with canonical DNA bases—adenine (A), thymine (T), guanine (G), and cytosine (C)—but also supports a variety of non-canonical and synthetic nucleobases. This functionality is especially powerful for researchers interested in exploring DNA sequences beyond the natural genetic code.

### Complementary Base Pairing Map

The complementary map includes both canonical and non-canonical bases, providing flexibility in designing sequences:

```python
complementary_map = {
    'A':'T', 'T':'A', 'G':'C', 'C':'G',
    'U':'A', 'D':'G', 'E':'T', 'L':'M',
    'M':'L', 'B':'S', 'S':'B', 'Z':'P',
    'P':'Z'
}
```

### Overview of Non-Canonical Bases
1. Uracil (U):
    Represents RNA incorporation into DNA, pairing with adenine (A).

3. tC (D):
    A tricyclic cytosine analogue, another fluorescent base known for its unique photophysical properties, pairs with guanine (G)~\cite{wilhelmsson2003photophysical}.

2. 2-Aminopurine (E):
    A fluorescent base analogue of adenine, used in studies requiring stable fluorescence, pairs with thymine (T)~\cite{ward1969fluorescence}.


4. Hydrophobic Base Pair (L and M):
    The hydrophobic pair, d5SICS (L) and dNaM (M), are examples of unnatural base pairs that maintain stability without hydrogen bonding, expanding the range of base pairing~\cite{malyshev2014semi}.

5. Hachimoji Bases (B-S and P-Z):
    These bases belong to the Hachimoji DNA system, which introduces four synthetic nucleotides that form orthogonal pairs: B pairs with S, and P pairs with Z. This system adds a new dimension of complexity to DNA structure and function~\cite{hoshika2019hachimoji}.

### Mutation and Customization

The `mutate()` function in mdna allows you to introduce these non-canonical bases into your DNA sequences. This capability enables the design of complex sequences for specific applications, such as studies involving fluorescence, hydrophobic interactions, or synthetic genetic systems.
Example of Non-Canonical Mutation

```python
# Load a DNA sequence with canonical bases
dna = mdna.make(sequence='AGCGATATAGA')

# Mutate the first base to a non-canonical base 'L' and the last base to 'P'
dna.mutate({0: 'L', dna.n_bp - 1: 'P'}, complementary=False)

# Describe the modified DNA structure to confirm changes
dna.describe()
```