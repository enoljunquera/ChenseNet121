import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from inceptionV3 import InceptionV3_Flexible
from prepare_LD50_data import prepare_dataset
from manage_3Ddescriptors import cargar_dataset_descriptores

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import argparse
parser = argparse.ArgumentParser(description="Evaluación de modelos InceptionV3 en conjunto de test.")
parser.add_argument('--model_dir', type=str, required=True, help='Ruta a la carpeta donde están los modelos fold1.pth a fold5.pth')
args = parser.parse_args()
model_dir = args.model_dir

# === Cargar datos ===
print("Cargando datos...")
df_test = pd.read_csv("./dataset/test.csv")
test_ids = df_test['Code'].tolist()

X, X_images, y = prepare_dataset('./dataset/combined_dataset.csv', include_images=True)

# === Cargar descriptores ===
descriptor_files = {
    '2x2x2_float': './dataset/dataset_descriptores_2x2x2_float.txt',
    '2x2x2_bool': './dataset/dataset_descriptores_2x2x2_bool.txt',
    'bool': './dataset/dataset_descriptores_bool.txt',
    'float': './dataset/dataset_descriptores_float.txt'
}
descriptor_data = {k: cargar_dataset_descriptores(v) for k, v in descriptor_files.items()}
all_ids = list(descriptor_data['float'].keys())

# === Filtrar test ===
final_data, final_images, final_labels = [], [], []
final_descriptors_2x2x2, final_descriptors_25x25x25, final_ids = [], [], []

for idx, id_value in enumerate(all_ids):
    if id_value in test_ids:
        final_data.append(X[idx])
        final_images.append(X_images[idx])
        final_labels.append(y[idx])
        final_ids.append(id_value)
        final_descriptors_2x2x2.append(np.concatenate([
            descriptor_data['2x2x2_bool'].get(id_value, np.zeros((2, 2, 2, 4))),
            descriptor_data['2x2x2_float'].get(id_value, np.zeros((2, 2, 2, 4)))
        ], axis=-1))
        final_descriptors_25x25x25.append(np.concatenate([
            descriptor_data['bool'].get(id_value, np.zeros((25, 25, 25, 4))),
            descriptor_data['float'].get(id_value, np.zeros((25, 25, 25, 4)))
        ], axis=-1))

# === Dataset personalizado ===
class LD50Dataset(Dataset):
    def __init__(self):
        self.data = final_data
        self.images = final_images
        self.labels = final_labels
        self.descriptors_small = final_descriptors_2x2x2
        self.descriptors_large = final_descriptors_25x25x25

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32).permute(2, 0, 1)
        volume_small = torch.tensor(self.descriptors_small[idx], dtype=torch.float32).permute(3, 0, 1, 2)
        volume_large = torch.tensor(self.descriptors_large[idx], dtype=torch.float32).permute(3, 0, 1, 2)
        descriptors = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, volume_small, volume_large, descriptors, label

test_dataset = LD50Dataset()
test_loader = DataLoader(test_dataset, batch_size=1)

# === Cargar modelos ===
models = []
for i in range(1, 6):
    model = InceptionV3_Flexible().to(device)
    model.load_state_dict(torch.load(f"{model_dir}/fold{i}.pth", map_location=device))
    model.eval()
    models.append(model)

# === Evaluación ===
rows = []
for i, (img, vol_s, vol_l, desc, label) in enumerate(test_loader):
    img, vol_s, vol_l, desc = img.to(device), vol_s.to(device), vol_l.to(device), desc.to(device)
    real = label.item()
    preds = []

    with torch.no_grad():
        for model in models:
            pred = model(image=img, volume_small=vol_s, volume_large=vol_l, descriptors=desc)
            preds.append(pred.item())

    avg = np.mean(preds)
    mae = abs(real - avg)
    mse = (real - avg) ** 2

    row = {
        "Core": final_ids[i],
        "real_oral_LD50": real,
        **{f"pred_fold{j+1}": preds[j] for j in range(5)},
        "pred_AVG": avg,
        "MAE": mae,
        "MSE": mse
    }
    rows.append(row)

# === Guardar CSV ===
df_result = pd.DataFrame(rows)
df_result.to_csv(model_dir + "/predicciones_inceptionV3_test.csv", index=False)
print("✅ Predicciones guardadas en " + model_dir + "/predicciones_inceptionV3_test.csv")
