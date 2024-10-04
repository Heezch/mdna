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
dna.mutate({0: 'L', dna.n_bp: 'P'}, complementary=False)

# Describe the modified DNA structure to confirm changes
dna.describe()
```