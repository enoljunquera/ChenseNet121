import os
import numpy as np
from prepare_LD50_data import prepare_dataset
from manage_3Ddescriptors import cargar_dataset_descriptores

# Definir rutas a los archivos de datos
dataset_path = './dataset/combined_dataset.csv'
descriptor_files = {
    '2x2x2_float': './dataset/dataset_descriptores_2x2x2_float.txt',
    '2x2x2_bool': './dataset/dataset_descriptores_2x2x2_bool.txt',
    'bool': './dataset/dataset_descriptores_bool.txt',
    'float': './dataset/dataset_descriptores_float.txt'
}

# Cargar datos de LD50 (numéricos + imágenes)
print("\n### Cargando datos de LD50 ###")
X, X_images, y = prepare_dataset(dataset_path, include_images=True)

# Cargar datos adicionales de descriptores 3D
print("\n### Cargando descriptores 3D ###")
descriptor_data = {key: {} for key in descriptor_files}
for key, file in descriptor_files.items():
    print(f"Cargando {key} desde {file}...")
    descriptor_data[key] = cargar_dataset_descriptores(file)

# Unir la información basada en los IDs disponibles en los descriptores
print("\n### Uniendo datos de LD50 y descriptores ###")
final_data = []
final_images = []
final_labels = []
final_descriptors_2x2x2_bool = []
final_descriptors_2x2x2_float = []
final_descriptors_bool = []
final_descriptors_float = []
final_smiles_descriptors = []

ids = list(descriptor_data['float'].keys())  # Se usa 'float' como referencia para los IDs

for idx, id_value in enumerate(ids):
    if id_value in descriptor_data['float']:
        final_data.append(X[idx])  # Incluye los 4 SMILES + 2 afinidades
        final_images.append(X_images[idx])
        final_labels.append(y[idx])
        final_descriptors_2x2x2_bool.append(descriptor_data['2x2x2_bool'].get(id_value, np.zeros((2,2,2,4))))
        final_descriptors_2x2x2_float.append(descriptor_data['2x2x2_float'].get(id_value, np.zeros((2,2,2,4))))
        final_descriptors_bool.append(descriptor_data['bool'].get(id_value, np.zeros((25,25,25,4))))
        final_descriptors_float.append(descriptor_data['float'].get(id_value, np.zeros((25,25,25,4))))
        final_smiles_descriptors.append(X[idx][:4])  # Extrae solo los 4 valores de SMILES
    else:
        print(f"Advertencia: No hay descriptores para ID {id_value}, omitiendo entrada.")

# Convertir a arrays numpy
final_data = np.array(final_data)
final_images = np.array(final_images)
final_labels = np.array(final_labels)
final_descriptors_2x2x2_bool = np.array(final_descriptors_2x2x2_bool)
final_descriptors_2x2x2_float = np.array(final_descriptors_2x2x2_float)
final_descriptors_bool = np.array(final_descriptors_bool)
final_descriptors_float = np.array(final_descriptors_float)
final_smiles_descriptors = np.array(final_smiles_descriptors)

# Mostrar información detallada sobre los datos
print("\n### Resumen final de los datos ###")
print(f"- Total de registros completos: {final_data.shape[0]} (con todos los datos disponibles)\n")

# Datos tabulares
print("Datos tabulares:")
print(f"  - Final_data (SMILES + afinidades): {final_data.shape} --> 4 descriptores SMILES + 2 afinidades")
print(f"  - Etiquetas (LD50): {final_labels.shape}")

# Imágenes
print("\nDatos de imágenes:")
print(f"  - Final_images: {final_images.shape} --> {final_images.shape[1:]} píxeles RGB\n")

# Descriptores 3D
print("Descriptores 3D:")
print(f"  - 2x2x2 bool: {final_descriptors_2x2x2_bool.shape}")
print(f"  - 2x2x2 float: {final_descriptors_2x2x2_float.shape}")
print(f"  - 25x25x25 bool: {final_descriptors_bool.shape}")
print(f"  - 25x25x25 float: {final_descriptors_float.shape}\n")

# Muestra de datos (para verificar)
print("\n### Ejemplo de los primeros 3 registros ###")
print("SMILES + Afinidades:")
print(final_data[:3])  # Muestra 3 ejemplos de los datos tabulares

print("\nLD50 (etiquetas):")
print(final_labels[:3])  # Muestra 3 etiquetas LD50

print("\nDescriptores SMILES:")
print(final_smiles_descriptors[:3])  # Muestra los valores SMILES de las primeras muestras
