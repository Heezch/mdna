
## Analyzing DNA Structures: Rigid Base Parameters and Visualization

In this section, we will analyze the DNA structure using rigid base parameters, which provide detailed information about the geometric properties of each base pair and base pair step. These parameters include shifts, tilts, rolls, and twists that describe the relative positioning and orientation of the bases within the DNA helix. The analysis will help us understand the structural dynamics and stability of DNA under various conditions.

### Overview of Rigid Base Parameters

Rigid base parameters are crucial for understanding DNA's mechanical properties. They describe the geometry of each base pair and the relative orientation between consecutive base pairs. These parameters are divided into two main categories:

#### **Base Pair Parameters**:
   - *Shear, Stretch, Stagger*: Describe the displacement of one base relative to the other within a base pair.
   - *Buckle, Propeller, Opening*: Describe the angular deformation within a base pair.

#### **Base Pair Step Parameters**:
   - *Shift, Slide, Rise*: Describe the relative displacement between two consecutive base pairs.
   - *Tilt, Roll, Twist*: Describe the angular relationship between two consecutive base pairs.

### Example: Analyzing DNA with Rigid Base Parameters

First, we load a pre-existing DNA structure from a trajectory file. For this example, assume the files are stored in an anonymized path.

```python
import mdna as mdna

# Load the DNA structure from the trajectory and topology files
dna = mdna.load(filename='/path_to_data/dry_0.xtc', top='/path_to_data/dry_0.pdb')

# Describe the DNA structure
dna.describe()
```

### Visualizing Rigid Base Parameters

We can plot the rigid base parameters to analyze the structural features of the DNA. This visualization includes confidence intervals for each parameter to indicate the range of possible variations.

```python
# Plot per base pair with confidence interval
rigid = dna.get_rigid_parameters()
_ = rigid.plot_parameters()
```

In the code block above, `plot_parameters()` generates plots for each rigid base parameter across the DNA sequence, with confidence intervals that provide insight into the variability and reliability of these measurements. These plots are crucial for identifying regions of the DNA that might have unusual or significant structural deviations.

### Distribution of Rigid Base Parameters

The distribution of each parameter is plotted below to provide insight into the variability and typical values for this DNA structure.

```python
# Plot distribution per parameter

params, names = rigid.get_parameters()  

import matplotlib as mpl, matplotlib.font_manager as font_manager
import seaborn as sns
import matplotlib.pyplot as plt

mpl.rcParams['font.family'] = 'serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif'] = cmfont.get_name()
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['axes.formatter.use_mathtext'] = True

colors = sns.color_palette("twilight", 12)
lims = [(-2,2),(-2,2),(-2,2),(-65,65),(-65,65),(-65,65)] + [(-3,3),(-3,3),(1.25,5),(-50,50),(-50,50),(0,60)]
fig, ax = plt.subplots(3, 4, figsize=(6, 2.5), sharey='col')
idx, jdx = 0, 0

for _, name in enumerate(names):
    para = params[:, 1:, names.index(name)]
    sns.kdeplot(para.flatten(), ax=ax[idx][jdx], fill=True, color='white', lw=5)
    sns.kdeplot(para.flatten(), ax=ax[idx][jdx], fill=True, color=colors[_], alpha=1, lw=1)
    ax[idx][jdx].set_title(name, x=0, y=0, color=colors[_], fontsize=10)
    ax[idx][jdx].set_xlim(lims[_])

    ax[idx][jdx].set_facecolor('none')
    ax[idx][jdx].set_yticklabels([])
    ax[idx][jdx].set_yticks([])
    ax[idx][jdx].set_ylabel('')

    if name in ['shear', 'buckle', 'shift', 'tilt', 'stretch', 'propeller']:
        ax[idx][jdx].set_xticks([])

    idx += 1
    if idx == 3:
        idx = 0
        jdx += 1
    if jdx == 4:
        jdx = 0

    if idx == 0 and jdx in [0, 1, 2, 3]:
        ax[idx][jdx].set_xticklabels([])
    if idx == 1 and jdx in [0, 1]:
        ax[idx][jdx].set_xticklabels([])

ax[-1][0].set_xlabel('[nm]', fontsize=11)
ax[-1][1].set_xlabel('[degrees]', fontsize=11)
ax[-1][2].set_xlabel('[nm]', fontsize=11)
ax[-1][3].set_xlabel('[degrees]', fontsize=11)
fig.tight_layout()
fig.subplots_adjust(hspace=-.25)

sns.despine(bottom=True, left=True)
```

In this section, we visualize the distributions of each rigid base parameter across the entire DNA sequence. By plotting the kernel density estimates (KDE) for each parameter, we can observe the typical values, variability, and any potential outliers. The customized appearance of the plots ensures that they are clear and suitable for presentation or publication.
