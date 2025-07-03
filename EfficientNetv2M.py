import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.efficientnet import efficientnet_v2_m

# Bloque EfficientNet adaptado a 3D
class EfficientBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, expansion=4):
        super(EfficientBlock3D, self).__init__()
        mid_channels = in_channels * expansion
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size=kernel_size, 
                               stride=stride, padding=padding, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_channels)
        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        return self.relu(x + identity)

# Modelo basado en EfficientNetV2-M
class EfficientNetV2MultiInput(nn.Module):
    def __init__(self, num_classes=1):
        super(EfficientNetV2MultiInput, self).__init__()
        
        # Modelo 2D basado en EfficientNetV2-M
        self.model_2d = efficientnet_v2_m(weights=None)
        self.model_2d.classifier[1] = nn.Linear(self.model_2d.classifier[1].in_features, 512)
        
        self.model_3d_small = nn.Sequential(
            EfficientBlock3D(8, 32),
            EfficientBlock3D(32, 64),
            EfficientBlock3D(64, 128),
            nn.AdaptiveAvgPool3d(1),  # Solo al final
            nn.Flatten(),
            nn.Linear(128, 64)
        )
        # Modelo 3D grande
        self.model_3d_large = nn.Sequential(
            EfficientBlock3D(8, 64),
            nn.MaxPool3d(2),
            EfficientBlock3D(64, 128),
            nn.MaxPool3d(2),
            EfficientBlock3D(128, 256),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(256, 128)
        )
        
        # Procesamiento de descriptores flotantes
        self.fc_descriptors = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Capa de fusión
        self.fusion_layer = nn.Sequential(
            nn.Linear(512 + 128 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Capa de salida
        self.output_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, image=None, volume_small=None, volume_large=None, descriptors=None):
        features = []
        
        if image is not None:
            features.append(self.model_2d(image))
        #if volume_small is not None:
        #    features.append(self.model_3d_small(volume_small))
        if volume_large is not None:
            features.append(self.model_3d_large(volume_large))
        if descriptors is not None:
            features.append(self.fc_descriptors(descriptors))
        
        if not features:
            raise ValueError("Se debe proporcionar al menos una entrada válida.")
        
        x = torch.cat(features, dim=1)
        x = self.fusion_layer(x)
        x = self.output_layer(x)
        return x
