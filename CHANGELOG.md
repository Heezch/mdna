# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

### Added
- `mdna.Nucleic`, `mdna.SplineFrames` and `mdna.BP_MAP` are public; `get_frames()` and `minimize()` document the frame layout and the Monte Carlo trace (`minimizer.out`).

### Changed
- Rescaling a spline to `n_bp` scales about the control points' centroid instead of the origin, so the structure stays where its control points are. Shapes centred on the origin are unchanged.

### Fixed
- `mutate(complementary=True)` raised a KeyError for residues on the second strand, and pairs shifted after non-canonical or methylated bases; residues now pair by chain position.
- `connect()` no longer depends on where the DNA sits: away from the origin the linker's base pairs were pulled apart (up to 80 A at 50 nm).
- NumPy 2 support (`np.NaN` in `GrooveAnalysis`).
- A debug print in `Mutate`.

## [0.1.1] - 2026-02-28

### Fixed
- Excluded accidental platform-specific compiled extension artifacts (`*.so`, `*.pyd`) and generated Cython build outputs from release artifacts.

## [0.1.0] - 2026-02-28

### Added
-

### Changed
- Major documentation refresh and structure cleanup for MkDocs content.
- Archived orphaned legacy pages and removed duplicate URL generation warnings.

### Fixed
- Resolved multiple primary URL warnings in site build output.
