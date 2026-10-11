# Corrected structure-grouped LD50 benchmark (October 2026)

**Status:** supplemental reproducibility material for the revised manuscript. The
earlier repository-root scripts and the original Zenodo v1 deposit are retained as
historical material; neither should be used by itself to reproduce the corrected
10-test results.

## Why a corrected dataset is supplied

The archived Zenodo v1 `combined_dataset.csv` contains **289 source records**, including
lactofen (Code 62276; oral LD50 59,600 mg/kg), whereas the later experimental
source used in this benchmark contains **288 records**. The later source corrects
the oral LD50 of propamocarb (Code 32490) from **43,500 to 5,250 mg/kg**.
We leave the Zenodo archive and the repository's historical `FlatDock_dataset`
unchanged to preserve provenance.

`dataset/combined_dataset.csv` is the corrected **288-record starting table**, not
the final evaluation cohort. The published source table requires identity and
modality curation. The audit returns **249 retained records / 199 unique structures**.
Full InChIKey (RDKit) or canonical isomeric SMILES fallback defines structural
grouping. Ambiguous identifiers, incompatible labels, and unavailable complete
modalities are excluded; multiple consistent records per structure are retained
and assigned to the same side of each partition.

## What is included

- `dataset/combined_dataset.csv`: corrected 288-record source (not Zenodo v1 data).
- `dataset/test_1_updated.csv` through `test_10_updated.csv`: original code-seeded
  split lists; effective test membership is expanded by unique structure.
- `dataset/voxel_features_nutridock_style_synthesized.csv`: 17 FlatDock features
  for 256 codes (the same table as in the historical repository).
- `audit/`: the resulting 249 retained records, 39 excluded records, and split
  composition (computed using the corrected CSV and Zenodo v1 modalities).
- `trainChenseNet121_full_grouped.py`, `densenet121.py`, `hyperparams.txt`:
  grouped training / five-fold checkpoint ensembles for ChenseNet121.
- `run_inception_10test_grouped.py`, `common.py`, `models/inceptionV3.py`:
  grouped training / five-fold checkpoint ensembles for ChInceptionV3.
- `baseline_no_docking_grouped.py`: descriptor-only classical baselines.
- `REENTRENAR_FLATDOCK_10TESTS_VERIFICABLE.py`: 17-feature FlatDock
  classical benchmarking; its legacy input tree can be reconstructed locally.
- `results_archives/`: corrected predictions, metrics, and split audits for CNNs,
  original classical baselines, and FlatDock variants. Prediction archives were
  generated during the manuscript revision; model checkpoint weights are not included.

## Getting the large input modalities

Download the archived `LD50_multimodal_dataset.zip` (Zenodo record 15800696)
and put it anywhere on disk. In this folder, run:

```shell
python preparar_dataset_local.py "C:\ruta\LD50_multimodal_dataset.zip"
python -m pip install -r requirements_revision_2026.txt
python baseline_no_docking_grouped.py --audit-only
```

The preparation script extracts both full-resolution and smaller voxel descriptor files, together with molecular images
from the original Zenodo archive; it **does not overwrite the corrected CSV**.
The audit (no training) should show 249 retained records, 199 unique structures,
and zero exact-structure overlap in all ten outer tests.

## Optional training/reproduction

From the `revised_2026` folder after preparing the modalities:

```shell
python trainChenseNet121_full_grouped.py --audit-only
python run_inception_10test_grouped.py --audit-only
# Full model retraining is computationally expensive and optional:
# python trainChenseNet121_full_grouped.py
# python run_inception_10test_grouped.py
```

FlatDock: reconstruct the canonical test reference files shipped as an archive,
then audit without retraining:

```shell
python preparar_flatdock_local.py
python REENTRENAR_FLATDOCK_10TESTS_VERIFICABLE.py --root flatdock_local --audit-only
```

For a complete rerun of the classical FlatDock regressors, pass
`--with-controls` (this creates new output files under `flatdock_local`).

## Evaluation and interpretation

Both CNNs use five structure-grouped training/validation folds per test,
validation-selected checkpoints, and mean ensemble predictions. Regression
performance is evaluated on **unique structures**: per-structure predictions
from repeated records are averaged before MSE/MAE calculation. The ten test
sets overlap, so their error summaries do not represent independent external
test cohorts. The primary endpoint is continuous acute oral LD50 regression;
exploratory classification at 2,000 mg/kg is not a separate optimized model.

The original Zenodo v1 dataset has not been silently altered. Any claims in
older code, tables, or metadata should be interpreted within their historical
experiment, not as corrected 2026 benchmark results.
