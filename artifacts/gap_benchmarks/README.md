# DNA gap benchmark source data

## 1WQY coordinates and derived suites

- Coordinate source: [RCSB PDB 1WQY](https://www.rcsb.org/structure/1WQY); downloaded from the official coordinate endpoint `https://files.rcsb.org/download/1wqy.pdb` on 2026-11-10.
- Primary citation: Arai et al., *Nucleic Acids Research* 33 (2005), 3017–3024, [doi:10.1093/nar/gki616](https://doi.org/10.1093/nar/gki616). RCSB classifies this as a 2.00 Å X-ray structure of a synthetic standard B-DNA decamer; its title records D2O crystallization, and the cited paper reports neutron-diffraction measurements with X-ray data.
- Local source: `1wqy_source.pdb`; SHA-256 `beb6100b7a1fb2e21989b73596c1f533aba8208b4e63ba6fbeb2640ded33a880`.
- Extracted library: `1wqy_dna_suites.json`; 14 internal standard A/C/G/T dinucleotide suites from chains A/B, covering 6 of 16 directed contexts. The two chains are one PDB entry and are not independent families.
- Reproduce with `scripts/gap_benchmarks/extract_1kx5_dna_suites.py --input artifacts/gap_benchmarks/1wqy_source.pdb --entry-id 1WQY --output artifacts/gap_benchmarks/1wqy_dna_suites.json`.

## 119D coordinates and derived suites

- Coordinate source: [RCSB PDB 119D](https://www.rcsb.org/structure/119D); downloaded from `https://files.rcsb.org/download/119d.pdb` on 2026-11-10.
- Primary citation: Leonard & Hunter, *Journal of Molecular Biology* 234 (1993), 198–208, [doi:10.1006/jmbi.1993.1574](https://doi.org/10.1006/jmbi.1993.1574). RCSB records a 2.25 Å X-ray structure of the standard synthetic B-DNA dodecamer `CGTAGATCTACG`, with two DNA chains and one partially occupied Mg²⁺; it is not a protein–DNA complex.
- Local source: `119d_source.pdb`; SHA-256 `ab465453c78dcd549d2ed7ac9bdac1af505452b3770c3f98ea6ab21274c86611`.
- Extracted library: `119d_dna_suites.json`; 18 internal standard A/C/G/T suites from chains A/B, covering 8 contexts, including previously absent `AG` and `CT`. The chains are one PDB entry and one crystallographic family, not independent replicates.
- Reproduce with `scripts/gap_benchmarks/extract_1kx5_dna_suites.py --input artifacts/gap_benchmarks/119d_source.pdb --entry-id 119D --output artifacts/gap_benchmarks/119d_dna_suites.json`.

## Four-source cross-entry panel

`1bna_3sv3_1wqy_119d_to_1kx5_dna_gap_panel.json` combines 18 1BNA, 17 3SV3, 14 1WQY, and 18 119D suites. The pool has 67 suites and all 16 directed standard DNA dinucleotide contexts. All nine fixed 1KX5 windows can be searched, but none yields a candidate under strict defaults, assembly-relaxed diagnostics, or early-gate sensitivity limits. The earlier three-source report is retained to show the prior two unsupported contexts. Full context coverage does not establish structural accuracy, independent-family generalization, or scientific acceptance.

## Base-frame versus full-anchor rigid fit

`scripts/gap_benchmarks/diagnose_dna_anchor_fit_modes.py` compares the current base-frame placements with independent Kabsch fits of each template suite to the full heavy-atom sets of the two observed 1KX5 anchors. It only uses the anchor residues for target coordinates; the middle-residue coordinates are not used to rank or generate candidates. The report is `dna_anchor_fit_comparison.json`.

Across 164 suite pairs in the nine windows, full-anchor fitting lowered the median maximum anchor-fit residual in every window, but no pair passed all existing geometry limits. The candidate gap residues from the independent fits disagree: all 164 pairs exceed at least one shared-residue geometry limit, and the first failed limit is middle-frame geometry for 163 pairs and shared-residue geometry for one pair. Pinned-bond and anchor-link checks also reject some pairs. This rules out a rigid fit alone as the closure fix; internal coordinates must be allowed to move or be jointly optimized. The diagnostic does not use masked truth, test environmental clashes, or establish scientific acceptance.

The existing WSL `md-env` OpenMM 8.3.1 environment loaded `amber14-all.xml` and created a parameterized System for a 119D DNA-only copy after adding hydrogens (24 residues, 760 particles); `implicit/gbn2.xml` also loads.

## Fixed-coordinate minimization backend probe

`scripts/gap_benchmarks/probe_amber_anchor_restraints.py` makes the backend and exact fixed-coordinate behavior reproducible without writing a model. On the local WSL `md-env` (OpenMM 8.3.1), it loaded the 119D standard-DNA source, added hydrogens, and parameterized 760 particles with Amber14/GBn2. With chain A residue 7 (DT; 32 particles) mobile and the other 728 particles assigned zero mass, a 100-iteration Reference run kept all fixed coordinates unchanged. The mobile heavy-atom maximum displacement was only 0.000189 nm, while potential energy rose from −17674.170 to −17670.484 kJ/mol. The CPU run differed from Reference by less than 0.004 kJ/mol. Repeating with the same Python random seed reproduced energies exactly.

This probe establishes backend availability and fixed-coordinate semantics only. The small motion and energy increase mean this setup has not yet demonstrated a useful local optimizer; it is not a gap-closure result or structural acceptance. No trajectory was run and no source coordinates were written.

RCSB data-use terms: [RCSB PDB Usage Policy](https://www.rcsb.org/pages/usage-policy).
