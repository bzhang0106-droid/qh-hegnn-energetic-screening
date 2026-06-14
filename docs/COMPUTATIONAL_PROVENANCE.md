# Computational provenance

The manuscript combines molecular generation, graph-neural-network prediction, semiempirical/DFT-derived descriptors, QTAIM analysis and post-screening validation.

## External software used by the full workflow

- ORCA for quantum-chemical calculations.
- xTB for semiempirical electronic descriptors and geometry-related features.
- Multiwfn and Critic2 for electrostatic and QTAIM post-processing.
- RDKit for molecular parsing, canonicalization and descriptor generation.
- PyTorch/PyTorch Geometric for graph-neural-network models.
- Slurm for remote HPC orchestration.

## Reviewer-facing reproducibility level

The repository supports lightweight reproducibility from processed tables. It does not bundle licensed software, raw wavefunction files, raw scratch directories, or binary model artifacts. Full end-to-end reproduction requires configuring the external programs above and rerunning the computational workflow on suitable hardware.

## npj computational reporting reminders

For electronic-structure results, the manuscript or supplementary materials should provide input parameters, basis sets and coordinates of input/output structures. For molecular dynamics, at least the initial and final configurations should be supplied. Any relative energies should be traceable to absolute energies in the supporting data.
