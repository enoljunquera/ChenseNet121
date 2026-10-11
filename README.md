# ChenseNet121: multimodal acute oral LD50 prediction

This repository contains research implementations and molecular-data resources
for continuous acute oral LD50 estimation with multimodal CNN architectures.

## Revised and audited benchmark (October 2026)

**Start with [`revised_2026/README.md`](revised_2026/README.md)** to reproduce
the corrected experiments reported in the revised manuscript. This includes:

- ChenseNet121 **and** ChInceptionV3, each evaluated on ten structure-grouped
  held-out partitions with five grouped validation folds per partition;
- 249 retained records representing 199 unique chemical structures, obtained
  by explicit identity/endpoint/modality curation of a corrected 288-record table;
- Ridge, LinearSVR and Random Forest baselines, including the 17-feature
  FlatDock representation comparison;
- matched structure/row predictions, group-split audits, and summary statistics.

### Dataset version distinction

The Zenodo v1 data archive (`10.5281/zenodo.15800696`) contains **289** source
records and retains the earlier propamocarb LD50 value of **43,500 mg/kg**.
The corrected 2026 experimental source contains **288** records, uses
**5,250 mg/kg** for propamocarb (Code 32490), and is provided in
[`revised_2026/dataset/combined_dataset.csv`](revised_2026/dataset/combined_dataset.csv).
The Zenodo archive additionally includes lactofen (Code 62276), not present
in the corrected experimental source. The curated evaluation subset (249/199)
is obtained by running the documented curation/audit, not by directly treating
either source file as the final test cohort.

The **old scripts in the repository root** (`trainDensenet121.py`,
`test_csv_densenet.py`, `prepare_LD50_data.py`, etc.) remain as
**historical implementations** and DO NOT reproduce the corrected
structure-grouped benchmark. Their default local `./dataset` folder and some
large input data are not included in the repository root; see the data
preparation instructions under `revised_2026/` instead. Files in
`FlatDock_dataset/` remain historical and should not be mistaken for the
corrected curated cohort. The ten original code-based lists in `Test_splits/`
are retained; expanded effective structure-grouped assignments are validated
in the revision scripts.

### Citation and usage

When citing the corrected manuscript, please use its final bibliographic
reference once published. The Zenodo v1 DOI identifies the original deposited
source, not the corrected 2026 experimental data. Results require further
independent evaluation before use in regulatory toxicological assessment.
