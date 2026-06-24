# LD50 Prediction with ChenseNet121

This repository provides code and resources for training and evaluating **ChenseNet121**, a flexible deep learning architecture based on DenseNet121 and adapted for **multimodal toxicity prediction**.

The model integrates different molecular representations, including:

- 2D molecular images
- 3D molecular descriptors, both as 2×2×2 and 25×25×25 volumes
- Numerical features derived from SMILES strings and docking affinities

The goal is to estimate **Acute Oral LD50** values of chemical compounds, supporting the development of in silico approaches that may contribute to reducing animal testing.

This repository also includes:

- The curated **FlatDock version of the dataset**, including the cleaned compound table and voxel-derived descriptor files.
- The compound code files corresponding to the **ten predefined train-test splits** used for repeated evaluation.
- Additional architecture files used for comparison with ChenseNet121.

---

## Project Structure

```text
├── densenet121.py                         # ChenseNet121 model architecture
├── trainDensenet121.py                    # 5-fold cross-validation training script
├── test_csv_densenet.py                   # External CSV-based test evaluation
├── main.py                                # Data exploration and visualization
├── prepare_LD50_data.py                   # Loads SMILES, affinities, and 2D images
├── manage_3Ddescriptors.py                # Loads 3D molecular descriptors
├── densenet_hyperparams.txt               # Hyperparameters for DenseNet-based models
├── hyperparams.txt                        # Additional configuration parameters
├── ChenseNet121_anaconda_environment.yaml # Conda environment file
├── additional_architectures.zip           # Additional architectures used for comparison
├── Test_splits/                           # Compound code files for the ten predefined train-test splits
└── FlatDock_dataset/                      # Curated FlatDock version of the dataset
    ├── combined_dataset.csv               # Main curated dataset
    ├── dataset_descriptores_float.txt     # 25×25×25 descriptors, float channels
    ├── dataset_descriptores_bool.txt      # 25×25×25 descriptors, Boolean channels
    ├── dataset_descriptores_2x2x2_float.txt # 2×2×2 descriptors, float channels
    ├── dataset_descriptores_2x2x2_bool.txt  # 2×2×2 descriptors, Boolean channels
    └── images/                            # 2D molecular images, if included or referenced
```

---

## Dataset

The curated dataset is provided in the `FlatDock_dataset/` folder. This folder contains the FlatDock version of the dataset used in the experiments, including the cleaned compound table, image references, docking-related variables, and voxel-derived descriptor files.

The `Test_splits/` folder contains the compound code files corresponding to the ten predefined train-test splits used for repeated evaluation.

| File or folder | Description |
|---|---|
| `FlatDock_dataset/combined_dataset.csv` | Main curated dataset with compound codes, SMILES, affinities, LD50 values, image paths, and related variables |
| `FlatDock_dataset/dataset_descriptores_float.txt` | 25×25×25 voxel-derived descriptors for continuous channels |
| `FlatDock_dataset/dataset_descriptores_bool.txt` | 25×25×25 voxel-derived descriptors for Boolean channels |
| `FlatDock_dataset/dataset_descriptores_2x2x2_float.txt` | 2×2×2 voxel-derived descriptors for continuous channels |
| `FlatDock_dataset/dataset_descriptores_2x2x2_bool.txt` | 2×2×2 voxel-derived descriptors for Boolean channels |
| `FlatDock_dataset/images/` | Folder containing 2D molecular images, if included or referenced by the dataset |
| `Test_splits/` | Compound code files for the ten predefined train-test splits |

---

## Installation

We recommend using Anaconda to set up the environment.

```bash
conda env create -f ChenseNet121_anaconda_environment.yaml
conda activate ld50env
```

Main requirements include:

- Python 3.9+
- PyTorch
- RDKit
- TensorFlow, used only for image preprocessing
- scikit-learn
- matplotlib
- pandas
- NumPy

Depending on the local configuration, some paths in the scripts may need to be updated to match the folder names used in this repository, especially the dataset path.

---

## How to Run

### 1. Training with Cross-Validation

Run 5-fold cross-validation and save the trained models:

```bash
python trainDensenet121.py
```

This script:

- Uses the configuration defined in `densenet_hyperparams.txt`
- Loads the multimodal input data
- Trains the ChenseNet121 architecture using cross-validation
- Saves trained models and outputs in the corresponding results folder

---

### 2. Testing on External CSV Files

The testing script evaluates trained models using a CSV file containing a `Code` column with the compound identifiers to be evaluated.

```bash
python test_csv_densenet.py
```

The predefined compound code files for the ten train-test splits are provided in the `Test_splits/` folder.

The script outputs a CSV file with the model predictions, typically named:

```text
predicciones_densenet121_test.csv
```

Before running the script, check that the model folder and test CSV path match the desired experiment.

---

### 3. Data Inspection

To explore SMILES strings, molecular images, docking affinities, and dataset content:

```bash
python main.py
```

This script can be used for basic data inspection and visualization.

---

## Model Overview

**ChenseNet121** extends DenseNet121 into a multimodal architecture for toxicity prediction. It combines several input branches:

- A 2D convolutional branch for molecular images
- A 3D convolutional branch for voxelized molecular descriptors
- A fully connected branch for tabular descriptors
- A late-fusion stage where the learned representations are concatenated and passed through dense layers

This design allows the model to jointly exploit structural, descriptor-based, and interaction-derived molecular information.

---

## Additional Architectures

The file `additional_architectures.zip` contains additional model architectures used for comparison with ChenseNet121.

These architectures were included to evaluate whether the proposed DenseNet-based multimodal approach provides competitive performance compared with other deep learning backbones.

---

## Reproducibility

To support reproducibility, this repository provides:

- The curated FlatDock dataset version used in the experiments
- The ten predefined train-test split code files
- The main ChenseNet121 training and testing scripts
- Hyperparameter configuration files
- The Conda environment file

The predefined test split files should be used when reproducing the repeated evaluation protocol.

---

## Reference

If you use this code or data, please cite:

> Junquera, E. et al. (2024). *SAVING MICE: ChenseNet121, a new deep learning architecture for LD50 toxicity estimation.*

---

## Notes

This repository is intended for research purposes. The predicted LD50 values should not be interpreted as direct replacements for regulatory toxicological assessment without further validation.

The main objective of this work is to explore deep learning-based toxicity prediction as a complementary in silico strategy to support safer and more ethical chemical risk assessment.
