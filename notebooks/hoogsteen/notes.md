# Hoogsteen Base Flip — Analysis Notes

## What the notebook does

A controlled mock trajectory is built by incrementally rotating a single purine
base from 0° to 360° around its glycosidic bond, collecting one structural frame
per step. This gives a smooth, deterministic scan of the full flip cycle. The
combined trajectory is then fed into the rigid-base parameter pipeline to observe
how all 12 base-pair and step parameters respond.

---

## The core problem: frame corruption by the Hoogsteen flip

### What the glycosidic rotation does to the Tsukuba frame

The Tsukuba base frame is constructed from three atoms: the glycosidic nitrogen
(N9 for purines), a ring carbon (C4), and the sugar C1'. A 180° rotation around
the C1'–N9 bond (syn/Hoogsteen conformation) has the following effect on the
derived frame vectors:

| Vector | Before flip | After flip | Reason |
|--------|------------|-----------|--------|
| `b_N` (normal) | up | **inverted** | C4 moves to opposite side → cross product reverses |
| `b_L` (long axis) | → major groove | **inverted** | rotation around `b_N` reverses when `b_N` flips |
| `b_D` (short axis) | unchanged | **unchanged** | double-negative in cross product cancels |

The net effect is a **180° rotation of the full frame around `b_D`**.

### Where the pipeline fails

`NucleicFrames.reshape_input()` applies an anti-parallel correction to strand B
before averaging the two frames into a mid-frame:

```python
rotation_B[:, [1, 2]] *= -1  # flips b_L and b_N of strand B only
```

This correction assumes all bases are in *anti* conformation. For a Hoogsteen
purine on strand A, `b_L` and `b_N` are already inverted, but **no correction is
applied to strand A**. The two frames passed to the mid-frame average are
geometrically inconsistent; the resulting mid-frame is rotated ~90° from any
physically meaningful orientation.

### Consequence cascade

```
Hoogsteen at position i
        │
        ├─► mid-frame at i is corrupted
        │       ├─► all intra-pair params at i are meaningless
        │       ├─► step params i−1 → i are artifacts
        │       └─► step params i → i+1 are artifacts
        │
        └─► all other positions: unaffected
```

---

## Detection: the arctan reaction coordinate

Rather than relying on any frame-based quantity, detection uses the two competing
hydrogen-bond distances that distinguish WC from Hoogsteen:

For an A·T pair:

$$\theta = \arctan\!\left(\frac{d_{N1_A - N3_T}}{d_{N7_A - N3_T}}\right)$$

| State | d_WC | d_HG | θ |
|-------|------|------|---|
| Watson-Crick | ~2.8 Å | ~5 Å | small (~30°) |
| Hoogsteen | ~5 Å | ~2.8 Å | large (~70°) |

Classification threshold: **θ = 45°**.  
θ also serves as a continuous reaction coordinate for monitoring the progression
of the flip — base classification is never binary in practice.

---

## Option 1: Frame-level correction (pre-computation)

For each base pair and each frame where θ > 45°, negate `b_L` and `b_N` of the
strand-A frame *before* passing to the parameter pipeline:

```
if θ(frame, i) > 45°:
    frames_A[i, :, 2] *= -1   # b_L
    frames_A[i, :, 3] *= -1   # b_N
```

**Effect:** Restores the frame to the convention the anti-parallel correction
expects. The mid-frame is now well-defined. All 12 parameters become physically
interpretable.

**Corrected values at a Hoogsteen site:**

| Parameter | Expected after correction |
|-----------|--------------------------|
| Shear | large positive — lateral displacement toward major groove |
| Opening | near ±180° — WC face of purine points away from complement |
| Twist at i±1 step | actual helical distortion, not artifact |
| Rise, Roll, Shift at i±1 | meaningful structural perturbation of neighbors |

**Implementation options:**

- **Post-hoc wrapper**: compute `base_frames` array from `NucleicFrames`, apply
  the mask, re-run `analyse_frames()`. Minimal code change, easy to prototype.
- **On-the-fly in `geometry.py`**: integrate θ detection inside
  `get_base_reference_frames()` and apply correction before storing frames.
  Cleaner, no redundant computation; requires touching core library code.

---

## Option 2: Post-hoc parameter-value correction

If frame-level correction is not practical, angular parameters can be
approximately corrected using the time trace and θ as a scaling axis.

**Works for:** buckle, propeller, opening, tilt, roll, twist  
The corruption of these parameters is approximately additive (a fixed offset that
scales with flip progress). For a fully flipped base:

$$\text{corrected}(t) = \text{observed}(t) - \Delta_\text{flip} \cdot \hat{\theta}(t)$$

where $\hat{\theta} = (\theta - \theta_{WC}) / (\theta_{HG} - \theta_{WC})$ and
$\Delta_\text{flip}$ is estimated directly from the trajectory (mean of corrupted
HG values minus the expected WC baseline).

**Does NOT work for:** shear, stretch, stagger, shift, slide, rise  
The translational parameters are projected onto the corrupted mid-frame axes.
The three components mix together in an orientation-dependent way; no scalar
offset can recover them.

---

## Recommended pipeline

```
for each frame t:
    1. compute θ(t, i) for all base pairs i          ← arctan of H-bond distances
    2. for each i where θ(t, i) > 45°:
           negate b_L and b_N of strand-A frame i     ← frame correction
    3. run NucleicFrames parameter computation         ← now valid everywhere
    4. store (params, θ) per frame

analysis:
    - plot any parameter vs. θ instead of frame index
    - θ gives a physically meaningful x-axis for all correlations
    - step params at i±1 now reveal true helical distortion from the flip
```

---

## Open questions / next steps

- [ ] Validate corrected parameters against reference Hoogsteen crystal structures
      (expected: shear ~+2 Å, opening ~180°)
- [ ] Test whether `fit_reference=True` in `NucleicFrames` interacts with the
      correction (it fits to canonical *anti* bases, so it may partially mask the
      issue or amplify it)
- [ ] For G·C pairs: Hoogsteen requires cytosine protonation (C⁺); decide whether
      to handle separately or flag as unsupported
- [ ] Decide on on-the-fly vs. wrapper implementation before merging into core