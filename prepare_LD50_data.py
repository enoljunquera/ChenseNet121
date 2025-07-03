import pandas as pd
import numpy as np
import os
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from rdkit.Chem import Descriptors
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def smiles_to_descriptors(smiles):
    """Convierte una cadena SMILES en un conjunto de descriptores moleculares."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return [
        Descriptors.MolWt(mol),
        Descriptors.TPSA(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumRotatableBonds(mol)
    ]

def load_images(image_paths, target_size=(300, 300)):
    """Carga imágenes y las convierte en arrays numpy."""
    images = []
    for path in image_paths:
        path = "./dataset" + path[1:]

        if os.path.exists(path):
            img = load_img(path, target_size=target_size, color_mode='rgb')
            img_array = img_to_array(img) / 255.0  # Normalizar
        else:
            img_array = np.zeros((*target_size, 3))  # Imagen vacía si no existe
        images.append(img_array)
    return np.array(images)

def prepare_dataset(file_path, include_images=False):
    """Carga y preprocesa el dataset."""
    df = pd.read_csv(file_path)
    
    # Selección de inputs
    descriptors = df['SMILES'].apply(smiles_to_descriptors)
    valid_rows = descriptors.notna()

    X_numeric = np.array(descriptors[valid_rows].tolist())
    X_affinity = df.loc[valid_rows, ['Affinity 7E3D', 'Affinity 8DT2']].values
    X = np.hstack([X_numeric, X_affinity])

    # Output (LD50 oral)
    y = df.loc[valid_rows, 'Acute Oral LD50'].values
  
    # Incluir imágenes si se solicita
    if include_images:
        X_images = load_images(df.loc[valid_rows, 'Image Path'].values)
        return X, X_images, y

    return X, y

# Uso del script

dataset_path = './dataset/combined_dataset.csv'
X, X_images, y = prepare_dataset(dataset_path, include_images=True)

# Separar un conjunto de prueba aleatorio para las características numéricas e imágenes
X_train, X_test, X_images_train, X_images_test, y_train, y_test = train_test_split(
    X, X_images, y, test_size=0.2, random_state=42
)

# Escalar los datos numéricos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Configurar validación cruzada
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def cross_validate(model, X, y):
    """Realiza validación cruzada con el modelo especificado e imprime detalles de las particiones."""
    fold = 1
    for train_index, val_index in kf.split(X):
        print(f"Fold {fold}: Train indices {train_index[:5]}... Val indices {val_index[:5]}...")
        fold += 1
    
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
    print(f'MAE promedio: {-scores.mean()}, desviación estándar: {scores.std()}')
    return scores

# Mostrar tamaños de los conjuntos
print(f"Tamaño del dataset total: {X.shape}, imágenes: {X_images.shape}")
print(f"Tamaño del conjunto de entrenamiento: {X_train.shape}, imágenes: {X_images_train.shape}, prueba: {X_test.shape}, imágenes: {X_images_test.shape}")