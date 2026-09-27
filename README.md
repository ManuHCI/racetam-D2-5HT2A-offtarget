# Off-target molecular dynamics of racetams at D2 and 5-HT2A

Simulation and analysis code for the manuscript on racetam off-target
engagement at the dopamine D2 and serotonin 5-HT2A receptors.

## Layout

    notebooks/d2        system preparation, docking and MD for the D2 systems
    notebooks/5ht2a     the same for 5-HT2A
    notebooks/sv2a      the SV2A on-target control
    analysis/           analysis scripts and ChimeraX rendering scripts
    data/mapping/       processed-to-canonical residue numbering tables
    data/logs/          OpenMM state logs for all production runs
    results/            per-replicate contact tables, figures and summary tables

## Simulations

Eighteen off-target trajectories: three racetams x two receptors x three
independent replicates, 50 ns each. Replicates branch from the same equilibrated
state with independently seeded velocities. Seed offsets: D2 levetiracetam
1000+n, D2 brivaracetam 2000+n, D2 piracetam 3000+n, 5-HT2A levetiracetam
4000+n, 5-HT2A brivaracetam 5000+n, 5-HT2A piracetam 6000+n, where n is the
replicate number.

## Preparation pipelines

Three pipelines were used, and they differ. This is described in the manuscript
Methods and is recorded here for reproducibility.

| System | Bilayer | Protein force field | Builder | Water |
|---|---|---|---|---|
| SV2A    | POPC | Amber ff14SB / Lipid17 | OpenMM Modeller | TIP3P-FB |
| D2      | DPPC | CHARMM36               | MemProtMD       | CHARMM TIP3P |
| 5-HT2A  | DPPC | Amber ff19SB / Lipid21 | packmol-memgen  | TIP3P |

Ligands were parameterised with OpenFF 2.1.0 (Sage) and NAGL/AM1-BCC charges in
all systems.

## Software

OpenMM 8.2, ParmEd 4.3.1, OpenFF Toolkit 0.18.0, OpenFF Interchange 0.5.2,
MDTraj, MDAnalysis, AutoDock Vina, Meeko, RDKit, UCSF ChimeraX 1.12.
Python 3.11. Simulations were run on a single NVIDIA GPU.

## Trajectories

Production trajectories (approximately 20 GB) are not included here. TODO: add
the data repository DOI.

## Citation

TODO: add once the paper is accepted.
