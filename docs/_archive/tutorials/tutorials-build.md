
## Advanced DNA Structure Generation and Manipulation with mdna

In this section, we'll explore more advanced functionalities of the `mdna` module, focusing on generating various DNA structures, modifying their topological properties, and using custom shapes. These tools allow you to create complex DNA configurations for detailed structural analysis.

### 1. Basic DNA Creation

You can create a DNA structure with no initial sequence, which will output a placeholder sequence ('DDD').

```python
# Build DNA with nothing, will output DDD sequence
dna = mdna.make()
dna.describe()
```

Alternatively, you can provide a specific sequence or define the number of base pairs (n_bp) to generate a random sequence.

```python
# Or provide a sequence
dna = mdna.make(sequence='GCGCGCGCGC')
dna.describe()

# Or provide a number of basepairs, resulting in a random sequence
dna = mdna.make(n_bp=10)
dna.describe()
```

### 2. Circular DNA and Topology Manipulation

The `mdna` module allows the creation of circular DNA (minicircles) and the manipulation of their topological properties, such as the linking number (Lk).

```python
# Or make a minicircle DNA in circular form
dna = mdna.make(n_bp=200, circular=True)
print('Lk, Wr, Tw', dna.get_linking_number())
dna.draw()

# Lets also minimize the DNA configuration
dna.minimize()

# See the final configuration
dna.draw()

# or save it to a file
dna.save('minimized_nbp_200_closed.pdb')
```

You can also change the linking number by under- or overwinding the DNA using the `dLk` parameter.

```python
# Also change the linking number by under or overwinding the DNA using the dLk parameter
dna = mdna.make(n_bp=200, circular=True, dLk=8)
dna.describe()
dna.get_linking_number()

# Minimize the DNA configuration, note to equilibrate the writhe use equilibrate_writhe=True, otherwise the Lk will not be conserved
dna.minimize(equilibrate_writhe=True)
dna.get_linking_number()
```

### 3. Using Custom Shapes

You can create DNA structures that follow custom shapes using the `Shape` class. For instance, you can generate DNA in the shape of a helix or define a completely custom path using control points.

```python
# We can also use custom shapes using the Shape class
control_points = mdna.Shapes.helix(height=3, pitch=5, radius=7, num_turns=4)
dna = mdna.make(n_bp=300, control_points=control_points)
dna.draw()

# Or use the control points to define a custom shape
control_points = np.array([[0,0,0],[30,10,-10],[50,10,20],[3,4,30]])
dna = mdna.make(n_bp=100, control_points=control_points, sequence=['A']*100)
dna.draw()
dna.describe()
dna.sequence
```

### 4. Extending DNA

The `extend` function allows you to add additional bases to an existing DNA structure, either in the forward or reverse direction.

```python
# We can also extend our DNA 
dna.extend(sequence=['G']*120)

# Or extend it in the opposite direction
dna.extend(sequence=['C']*120, forward=False)
dna.draw()
```

### 5. Connecting DNA Strands

You can generate two separate strands of DNA and connect them to form a continuous double helix. The `connect` function optimizes the connection to minimize the twist between the strands.

```python
# Lets generate two strands of DNA and displace the second one away from the first one
dna0 = mdna.make(sequence='AAAAAAAAA', control_points=mdna.Shapes.line(1))
dna1 = mdna.make(sequence='GGGGGGGGG', control_points=mdna.Shapes.line(1) + np.array([4,0,-5]))

# Now we can connect the two strands, the function will find the optimal number of basepairs to connect the two strands to minimize the twist 
dna2 = mdna.connect(dna0, dna1)
dna2.draw()
dna2.describe()
```

### 6. Visualization

For advanced visualization, you can use `nglview` to visualize the Monte Carlo minimized configuration of the DNA structure.

```python
# visualize using nglview MC minimization
view = nv.show_mdtraj(dna2.get_MC_traj())
view
```

This concludes the advanced tutorial on generating, manipulating, and visualizing DNA structures with the `mdna` module. These examples illustrate the flexibility and power of the module in creating detailed and customized DNA configurations for a wide range of research applications.
