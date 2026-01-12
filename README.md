
# LD50 Prediction with ChenseNet121

This repository provides code and resources for training and evaluating **ChenseNet121**, a flexible deep learning architecture based on DenseNet121, adapted for **multimodal toxicity prediction**. The model integrates:

-  2D molecular images  
-  3D molecular descriptors (both 2×2×2 and 25×25×25 volumes)  
-  Numerical features from SMILES strings and docking affinities  

The goal is to estimate the **Acute Oral LD50** values of chemical compounds, aiding in the reduction of animal testing through in silico methods.

---

##  Project Structure

```
├── densenet121.py                  # ChenseNet121 model architecture
├── trainDensenet121.py            # 5-fold training script
├── test_csv_densenet.py           # External test set evaluation
├── main.py                        # Data exploration and visualization
├── prepare_LD50_data.py           # Loads SMILES, affinities, and 2D images
├── manage_3Ddescriptors.py        # Loads 3D molecular descriptors
├── densenet_hyperparams.txt       # Model training hyperparameters
├── environment.yml                # [Add your Conda environment file here]
└── dataset/
    ├── combined_dataset.csv                         # Main dataset
    ├── test.csv                                     # External test codes
    ├── dataset_descriptores_float.txt               # 25x25x25 descriptors (float)
    ├── dataset_descriptores_bool.txt                # 25x25x25 descriptors (bool)
    ├── dataset_descriptores_2x2x2_float.txt         # 2x2x2 descriptors (float)
    ├── dataset_descriptores_2x2x2_bool.txt          # 2x2x2 descriptors (bool)
    └── [image folder referenced in combined_dataset.csv]
```

---

##  Installation

We recommend using [Anaconda](https://www.anaconda.com/) to set up the environment:

```bash
conda env create -f environment.yml
conda activate ld50env
```

**Requirements include:**
- Python 3.9+
- PyTorch
- RDKit
- TensorFlow (used only for image preprocessing)
- scikit-learn
- matplotlib

---

##  How to Run

### 1. Training with Cross-Validation

Run 5-fold cross-validation and save the trained models:

```bash
python trainDensenet121.py
```

- Uses the settings from `densenet_hyperparams.txt`
- Outputs models to `results/densenet121/modelX/` (where X is the next available ID)

---

### 2. Testing on External CSV

Requires a file `./dataset/test.csv` with a column `Code` listing compound IDs.

```bash
python test_csv_densenet.py
```

- Ensure the path in the script matches your desired model folder (default: `results/densenet121/modelX_test_csv/`)
- Outputs predictions as `predicciones_densenet121_test.csv`

---

### 3. Data Inspection

To explore SMILES, images, and affinities:

```bash
python main.py
```

---

##  Dataset Structure

Place all required data in the `./dataset/` folder:

| File | Description |
|------|-------------|
| `combined_dataset.csv` | Main dataset with SMILES, affinities, LD50, image paths, and compound codes |
| `test.csv` | List of compound IDs to test |
| `dataset_descriptores_*.txt` | 3D descriptors used as volumetric data |
| `images/` | Folder containing 2D molecular images |

---

##  Reference

If you use this code or data, please cite:

> Junquera, E. et al. (2024). *SAVING MICE: ChenseNet121 a new deep learning architecture for LD50 Toxycity Estimation.*

---

##  About the Model

**ChenseNet121** extends DenseNet121 with:
- 2D convolutional branches for RGB images  
- 3D convolutional branches for voxelized molecular descriptors  
- A fully connected branch for tabular descriptors  
- Merged via concatenation followed by dense layers  

Trained on a hybrid dataset derived from [Junquera et al., 2024], this model achieves robust performance in LD50 regression and sets the basis for future toxicogenomics applications.

---
