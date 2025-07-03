import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from inceptionV3 import InceptionV3_Flexible
from prepare_LD50_data import prepare_dataset
from manage_3Ddescriptors import cargar_dataset_descriptores


# Leer hiperparámetros desde un archivo externo
def load_hyperparams(filepath):
    hyperparams = {}
    with open(filepath, 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            hyperparams[key.strip()] = float(value.strip())
    return hyperparams

hyperparams = load_hyperparams('hyperparams.txt')
learning_rate = hyperparams['learning_rate']
num_epochs = int(hyperparams['epochs'])
batch_size = int(hyperparams['batch_size'])
k_folds = 5

# Crear directorio de resultados
if not os.path.exists('results/inceptionV3'):
    os.makedirs('results/inceptionV3')

model_id = len(os.listdir('results/inceptionV3')) + 1
save_path = f'results/inceptionV3/modelo{model_id}'
os.makedirs(save_path)
shutil.copy('hyperparams.txt', f'{save_path}/hyperparams.txt')

# Cargar datos de LD50
print("\n### Cargando datos de LD50 ###")
X, X_images, y = prepare_dataset('./dataset/combined_dataset.csv', include_images=True)

# Cargar descriptores 3D
descriptor_files = {
    '2x2x2_float': './dataset/dataset_descriptores_2x2x2_float.txt',
    '2x2x2_bool': './dataset/dataset_descriptores_2x2x2_bool.txt',
    'bool': './dataset/dataset_descriptores_bool.txt',
    'float': './dataset/dataset_descriptores_float.txt'
}

descriptor_data = {key: cargar_dataset_descriptores(file) for key, file in descriptor_files.items()}

# Unir datos en una estructura adecuada (NO SE TOCA)
final_data, final_images, final_labels, final_descriptors_2x2x2, final_descriptors_25x25x25 = [], [], [], [], []
ids = list(descriptor_data['float'].keys())

for idx, id_value in enumerate(ids):
    if id_value in descriptor_data['float']:
        final_data.append(X[idx])
        final_images.append(X_images[idx])
        final_labels.append(y[idx])
        final_descriptors_2x2x2.append(np.concatenate([
            descriptor_data['2x2x2_bool'].get(id_value, np.zeros((2,2,2,4))),
            descriptor_data['2x2x2_float'].get(id_value, np.zeros((2,2,2,4)))
        ], axis=-1))
        final_descriptors_25x25x25.append(np.concatenate([
            descriptor_data['bool'].get(id_value, np.zeros((25,25,25,4))),
            descriptor_data['float'].get(id_value, np.zeros((25,25,25,4)))
        ], axis=-1))

# Convertir a arrays numpy
final_data = np.array(final_data)
final_images = np.array(final_images)
final_labels = np.array(final_labels)
final_descriptors_2x2x2 = np.array(final_descriptors_2x2x2)
final_descriptors_25x25x25 = np.array(final_descriptors_25x25x25)

# Definir dataset personalizado (NO SE TOCA)
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
        label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)
        return image, volume_small, volume_large, descriptors, label

# Crear dataset y dividir en entrenamiento y test
dataset = LD50Dataset()
test_size = int(0.2 * len(dataset))
train_size = len(dataset) - test_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Configuración de K-Fold
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(train_dataset)))):
    print(f"\n### Fold {fold + 1}/{k_folds} ###")
    train_subsampler = torch.utils.data.Subset(train_dataset, train_idx)
    val_subsampler = torch.utils.data.Subset(train_dataset, val_idx)
    train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subsampler, batch_size=batch_size)
    
    model = InceptionV3_Flexible().to("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Lista para almacenar la pérdida de validación
    val_losses = []

    # Agregar esto antes del bucle de entrenamiento
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=100, verbose=True)

    # Inicializar la mejor pérdida de validación en cada fold
    best_val_loss = float("inf")
    best_model_path = f"{save_path}/fold{fold+1}.pth"

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for image, volume_small, volume_large, descriptors, label in train_loader:
            image, volume_small, volume_large, descriptors, label = (
                image.to("cuda"), volume_small.to("cuda"), volume_large.to("cuda"), 
                descriptors.to("cuda"), label.to("cuda")
            )
            optimizer.zero_grad()
            outputs = model(image=image, volume_small=volume_small, volume_large=volume_large, descriptors=descriptors)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Fold {fold+1}, Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

        # Evaluación en validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for image, volume_small, volume_large, descriptors, label in val_loader:
                image, volume_small, volume_large, descriptors, label = (
                    image.to("cuda"), volume_small.to("cuda"), volume_large.to("cuda"), 
                    descriptors.to("cuda"), label.to("cuda")
                )
                outputs = model(image=image, volume_small=volume_small, volume_large=volume_large, descriptors=descriptors)
                loss = criterion(outputs, label)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Fold {fold+1}, Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")

        # Guardar solo si mejora
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Nuevo mejor modelo guardado para Fold {fold+1} en {best_model_path}")

        # Aplicar el scheduler para reducir el LR si no mejora
        scheduler.step(val_loss)

        val_losses.append(val_loss)


    # Guardar modelo
    torch.save(model.state_dict(), f"{save_path}/fold{fold + 1}.pth")

    # Generar y guardar la gráfica de validación
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), val_losses, marker='o', linestyle='-', color='b', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Validation Loss - Fold {fold+1}')
    plt.legend()
    plt.grid()
    plt.savefig(f"{save_path}/fold{fold+1}_loss.png")
    plt.close()


# Cargar modelos entrenados en una lista
model_paths = [f"{save_path}/fold{i+1}.pth" for i in range(k_folds)]
models = []

for path in model_paths:
    model = InceptionV3_Flexible().to("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path))
    model.eval()
    models.append(model)

# Evaluar en test con el ensamble de modelos
test_loss = 0.0
num_models = len(models)
predictions = []
true_values = []

with torch.no_grad():
    for image, volume_small, volume_large, descriptors, label in test_loader:
        image, volume_small, volume_large, descriptors, label = (
            image.to("cuda"), volume_small.to("cuda"), volume_large.to("cuda"), descriptors.to("cuda"), label.to("cuda")
        )
        
        # Hacer predicción con cada modelo y promediar
        preds = torch.stack([m(image=image, volume_small=volume_small, volume_large=volume_large, descriptors=descriptors) for m in models])
        mean_pred = preds.mean(dim=0)

        loss = criterion(mean_pred, label)
        test_loss += loss.item()
        
        predictions.extend(mean_pred.cpu().numpy())
        true_values.extend(label.cpu().numpy())

test_loss /= len(test_loader)
print(f"Final Test Loss (Mean of Models): {test_loss:.4f}")


# Ruta del archivo para guardar los resultados
results_file = f"{save_path}/mse_mae_results.txt"

# Evaluación en validación y test
val_mse_list, val_mae_list = [], []

# Evaluación en cada fold
for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(train_dataset)))):

    print(f"\n### Evaluando Fold {fold+1} ###")

    # Cargar el modelo correspondiente al fold
    model_path = f"{save_path}/fold{fold+1}.pth"
    model = InceptionV3_Flexible().to("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Crear el loader de validación específico de este fold
    val_subsampler = torch.utils.data.Subset(train_dataset, val_idx)
    val_loader = DataLoader(val_subsampler, batch_size=batch_size)

    val_true = []
    val_pred = []

    with torch.no_grad():
        for image, volume_small, volume_large, descriptors, label in val_loader:
            image, volume_small, volume_large, descriptors, label = (
                image.to("cuda"), volume_small.to("cuda"), volume_large.to("cuda"), descriptors.to("cuda"), label.to("cuda")
            )

            outputs = model(image=image, volume_small=volume_small, volume_large=volume_large, descriptors=descriptors)

            val_true.extend(label.cpu().numpy())
            val_pred.extend(outputs.cpu().numpy())

    # Calcular MSE y MAE para la validación en este fold
    val_mse = mean_squared_error(val_true, val_pred)
    val_mae = mean_absolute_error(val_true, val_pred)

    val_mse_list.append(val_mse)
    val_mae_list.append(val_mae)

    print(f"Fold {fold+1} - Validation MSE: {val_mse:.4f}, Validation MAE: {val_mae:.4f}")

# Evaluación final en test
test_loss = 0.0
predictions = []
true_values = []

with torch.no_grad():
    for image, volume_small, volume_large, descriptors, label in test_loader:
        image, volume_small, volume_large, descriptors, label = (
            image.to("cuda"), volume_small.to("cuda"), volume_large.to("cuda"), descriptors.to("cuda"), label.to("cuda")
        )
        
        preds = torch.stack([m(image=image, volume_small=volume_small, volume_large=volume_large, descriptors=descriptors) for m in models])
        mean_pred = preds.mean(dim=0)

        loss = criterion(mean_pred, label)
        test_loss += loss.item()
        
        predictions.extend(mean_pred.cpu().numpy())
        true_values.extend(label.cpu().numpy())

# Cálculo de MSE y MAE en test
test_mse = mean_squared_error(true_values, predictions)
test_mae = mean_absolute_error(true_values, predictions)

# Guardar los resultados en un archivo txt
with open(results_file, "w") as f:
    f.write(f"Validation MSE por fold: {val_mse_list}\n")
    f.write(f"Validation MAE por fold: {val_mae_list}\n")
    f.write(f"Final Test MSE: {test_mse}\n")
    f.write(f"Final Test MAE: {test_mae}\n")

print(f"Resultados guardados en {results_file}")
