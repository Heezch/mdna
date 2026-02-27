# Getting Started with MDNA

Welcome to the MDNA toolkit! This section will guide you through the basics of using MDNA to create and analyze DNA structures. By the end of this guide, you'll have written your first lines of code, generated a DNA structure, and performed basic analysis—all with just a few simple steps.

## Step 1: Installation

Before you can start using MDNA, you need to install it. MDNA is a Python library, so you can install it using pip. Open your terminal and run the following command:

```bash
pip install mdna
```

or dircetly clone the github repository
```bash
git clone --recurse-submodules -j8 git@github.com:heezch/mdna.git
```

Once the installation is complete, you’re ready to start coding.



## Step 2: Generating a DNA Structure

Let's dive right in by creating a simple DNA structure. We’ll use MDNA to generate a double-stranded DNA with a custom sequence.

```python
import mdna as mdna

# Generate a DNA structure from a given sequence
dna = mdna.make(sequence="ATCGATCGGT")

# View basic information about the DNA structure
dna.describe()

# or quick draw the structure
dna.draw()
```


## Step 3: Minimizing the Structure

Next, we’ll optimize the structure to ensure it's physically realistic. This is done through energy minimization.

```python
# Minimize the energy of the DNA structure
dna.minimize()
```

Note that by using the `minimize()` method the traj instance is updated internally. 


## Step 4: Analyzing the Structure

Now that we have a minimized DNA structure, let's perform some basic analysis. We’ll calculate the rigid base parameters, which describe the relative positions of the base pairs.


```python
# Compute rigid base parameters
params, names = dna.get_parameters()

# Output the results
print("Rigid base parameters calculated:", params.shape)
```

This step gives you a deeper understanding of the DNA’s structural dynamics. The parameters calculated here are essential for more advanced studies, but don’t worry—just getting this far is a great start!


## Conclusion: You've Done It!

Congratulations! You've just created, minimized, analyzed, and visualized a DNA structure using the MDNA toolkit. With these basic steps, you're well on your way to exploring the full capabilities of MDNA. Remember, the key to mastering this toolkit is practice—try modifying the sequence, creating circular DNA, or analyzing different properties to see what happens.

Now that you’ve gotten started, you can dive deeper into more advanced features of MDNA. But for now, take a moment to appreciate what you've achieved. You've not only learned how to use a powerful scientific tool, but you've also taken your first steps into the world of molecular dynamics and DNA analysis.
