# Rigid Base Parameters

This page explains the rigid base formalism used by MDNA for describing DNA geometry.

---

## Overview

The rigid base model treats each nucleobase as a rigid body with a well-defined reference frame. DNA geometry is then described by the relative positions and orientations of these frames, yielding 12 parameters per base pair step.

---

## Reference Frames

Each base has a local coordinate system (reference frame) defined by the **Tsukuba convention**:

- **Origin** ($\mathbf{b}_R$): The base reference point, computed from the glycosidic nitrogen and C1' sugar atom
- **Long axis** ($\hat{b}_L$): Points roughly along the base pair hydrogen bonds
- **Short axis** ($\hat{b}_D$): Perpendicular to the long axis, in the base plane
- **Normal** ($\hat{b}_N$): Perpendicular to the base plane (right-hand rule)

The mid-step frame (average of two consecutive base pair frames) serves as the reference for computing step parameters.

---

## Base Pair Parameters

These describe the relative geometry of two bases **within** a Watson-Crick pair:

| Parameter | Description | Axis | Unit |
|-----------|-------------|------|------|
| **Shear** | Lateral displacement | Along $\hat{b}_L$ | nm |
| **Stretch** | Separation along H-bond direction | Along $\hat{b}_D$ | nm |
| **Stagger** | Vertical offset | Along $\hat{b}_N$ | nm |
| **Buckle** | Rotation opening the base pair like a book | Around $\hat{b}_L$ | degrees |
| **Propeller** | Rotation of bases in opposite directions | Around $\hat{b}_D$ | degrees |
| **Opening** | Rotation that opens the Watson-Crick edge | Around $\hat{b}_N$ | degrees |

---

## Base Pair Step Parameters

These describe the relative geometry **between** consecutive base pair steps:

| Parameter | Description | Axis | Unit |
|-----------|-------------|------|------|
| **Shift** | Lateral displacement of one step relative to the next | Along $\hat{b}_L$ | nm |
| **Slide** | Displacement along the short axis | Along $\hat{b}_D$ | nm |
| **Rise** | Vertical separation between steps | Along $\hat{b}_N$ | nm |
| **Tilt** | Rotation around the long axis | Around $\hat{b}_L$ | degrees |
| **Roll** | Rotation around the short axis (bending) | Around $\hat{b}_D$ | degrees |
| **Twist** | Rotation around the helical axis | Around $\hat{b}_N$ | degrees |

---

## Computation Method

MDNA computes these parameters through the following steps:

1. **Base identification**: Extract nucleobase heavy atoms from the MDTraj trajectory
2. **Frame fitting**: Fit each base to a canonical reference frame using the `ReferenceBase` class
3. **Mid-pair frames**: Average the two base frames within each pair
4. **Euler decomposition**: Compute the rotation matrix and displacement between consecutive mid-pair frames
5. **Parameter extraction**: Decompose the rotation into Euler angles (Tilt, Roll, Twist) and project the displacement onto the local axes (Shift, Slide, Rise)

The rotation decomposition uses the `RigidBody.extract_omega_values()` method, which handles edge cases near $\pm\pi$ rotation angles.

---

## Typical Values for B-DNA

| Parameter | Typical Range |
|-----------|---------------|
| Shift | -0.5 to 0.5 nm |
| Slide | -0.5 to 0.5 nm |
| Rise | 0.31 to 0.37 nm |
| Tilt | -10° to 10° |
| Roll | -10° to 10° |
| Twist | 30° to 40° (mean ~36°) |
| Shear | -0.5 to 0.5 nm |
| Stretch | -0.3 to 0.3 nm |
| Stagger | -0.5 to 0.5 nm |
| Buckle | -20° to 20° |
| Propeller | -25° to -5° |
| Opening | -5° to 5° |

---

## References

- Olson, W. K., et al. (2001). A standard reference frame for the description of nucleic acid base-pair geometry. *J. Mol. Biol.*, 313(1), 229–237.
- Lu, X. J., & Olson, W. K. (2003). 3DNA: a software package for the analysis, rebuilding and visualization of three-dimensional nucleic acid structures. *Nucleic Acids Res.*, 31(17), 5108–5121.
