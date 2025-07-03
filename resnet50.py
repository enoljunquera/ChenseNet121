import torch
import torch.nn as nn
import torch.nn.functional as F

# Bloque Bottleneck para ResNet-50 en 3D
class Bottleneck3D(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(Bottleneck3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_channels)
        
        self.conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(mid_channels)
        
        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        return F.relu(out)
        
# Bloque Bottleneck para ResNet-50 en 2D
class Bottleneck2D(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super(Bottleneck2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        return F.relu(out)

# Modelo ResNet50 Flexible con soporte para inputs 2D, 3D y descriptores
class ResNet50_Flexible(nn.Module):
    def __init__(self, num_descriptors=6):
        super(ResNet50_Flexible, self).__init__()
        
        # Modelo 2D basado en ResNet-50
        self.model_2d = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Capas de ResNet-50 con bloques Bottleneck
            Bottleneck2D(64, 64, 256, stride=1),
            Bottleneck2D(256, 64, 256, stride=1),
            Bottleneck2D(256, 64, 256, stride=1),

            Bottleneck2D(256, 128, 512, stride=2),
            Bottleneck2D(512, 128, 512, stride=1),
            Bottleneck2D(512, 128, 512, stride=1),
            Bottleneck2D(512, 128, 512, stride=1),

            Bottleneck2D(512, 256, 1024, stride=2),
            Bottleneck2D(1024, 256, 1024, stride=1),
            Bottleneck2D(1024, 256, 1024, stride=1),
            Bottleneck2D(1024, 256, 1024, stride=1),
            Bottleneck2D(1024, 256, 1024, stride=1),
            Bottleneck2D(1024, 256, 1024, stride=1),

            Bottleneck2D(1024, 512, 2048, stride=2),
            Bottleneck2D(2048, 512, 2048, stride=1),
            Bottleneck2D(2048, 512, 2048, stride=1),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, 128)  # Cambio de 64 → 2048 para ajustarse a ResNet-50
        )

        
        # Modelo 3D Pequeño
        self.model_3d_small = self._make_resnet_3d(8, 64)
        
        # Modelo 3D Grande
        self.model_3d_large = self._make_resnet_3d(8, 128)
        
        # Capa para descriptores
        self.fc_descriptors = nn.Sequential(
            nn.Linear(num_descriptors, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),  # Capa añadida para igualar DenseNet
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.Dropout(0.3)  # Dropout para evitar sobreajuste
        )
        
        # Fusión de características
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 + 128 + 64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128)  # Capa adicional para igualar DenseNet
        )
        
        # Capa de salida para regresión
        self.output_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),  # Capa adicional para igualar DenseNet
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        
    def _make_resnet_3d(self, in_channels, out_features):
        return nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),

            # Capa 1 (3 bloques)
            Bottleneck3D(64, 64, 256, stride=1),
            Bottleneck3D(256, 64, 256, stride=1),
            Bottleneck3D(256, 64, 256, stride=1),

            # Capa 2 (4 bloques)
            Bottleneck3D(256, 128, 512, stride=2),
            Bottleneck3D(512, 128, 512, stride=1),
            Bottleneck3D(512, 128, 512, stride=1),
            Bottleneck3D(512, 128, 512, stride=1),

            # Capa 3 (6 bloques)
            Bottleneck3D(512, 256, 1024, stride=2),
            Bottleneck3D(1024, 256, 1024, stride=1),
            Bottleneck3D(1024, 256, 1024, stride=1),
            Bottleneck3D(1024, 256, 1024, stride=1),
            Bottleneck3D(1024, 256, 1024, stride=1),
            Bottleneck3D(1024, 256, 1024, stride=1),

            # Capa 4 (3 bloques)
            Bottleneck3D(1024, 512, 2048, stride=2),
            Bottleneck3D(2048, 512, 2048, stride=1),
            Bottleneck3D(2048, 512, 2048, stride=1),

            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(2048, out_features)
        )

        
    def forward(self, image=None, volume_small=None, volume_large=None, descriptors=None):
        features = []
        
        if image is not None:
            features.append(self.model_2d(image))
        """
        if volume_small is not None:
            features.append(self.model_3d_small(volume_small))
        """
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
