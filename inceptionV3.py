import torch
import torch.nn as nn

# Bloque Inception mejorado (basado en InceptionV3 oficial)
class InceptionBlock2D(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj):
        super(InceptionBlock2D, self).__init__()

        # 1x1 Convolution branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )

        # 1x1 -> (1x3 + 3x1) Convolution branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3, ch3x3, kernel_size=(3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )

        # 1x1 -> (3x3 + 3x3) Convolution branch
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5, ch5x5, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True)
        )

        # Pooling branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1, bias=False),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)
        
# Bloque Inception para volúmenes 3D
class InceptionBlock3D(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj):
        super(InceptionBlock3D, self).__init__()

        # 1x1 Convolution branch
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, ch1x1, kernel_size=1, bias=False),
            nn.BatchNorm3d(ch1x1),
            nn.ReLU(inplace=True)
        )

        # 1x1 -> (1x3x3 + 3x1x1) Convolution branch
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, ch3x3_reduce, kernel_size=1, bias=False),
            nn.BatchNorm3d(ch3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch3x3_reduce, ch3x3, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(ch3x3),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch3x3, ch3x3, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(ch3x3),
            nn.ReLU(inplace=True)
        )

        # 1x1 -> (3x3x3 + 3x3x3) Convolution branch
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, ch5x5_reduce, kernel_size=1, bias=False),
            nn.BatchNorm3d(ch5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch5x5_reduce, ch5x5, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(ch5x5),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch5x5, ch5x5, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(ch5x5),
            nn.ReLU(inplace=True)
        )

        # Pooling branch
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, pool_proj, kernel_size=1, bias=False),
            nn.BatchNorm3d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)



# Modelo Inception Flexible (2D + 3D + Descriptores)
class InceptionV3_Flexible(nn.Module):
    def __init__(self, in_channels_2d=3, in_channels_3d=8, num_descriptors=6):
        super(InceptionV3_Flexible, self).__init__()

        # Modelo para imágenes 2D con Inception mejorado
        self.model_2d = nn.Sequential(
            nn.Conv2d(in_channels_2d, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionBlock2D(64, 64, 48, 64, 16, 32, 32),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(192, 128)
        )

        ch1x1 = 8
        ch3x3 = 64
        ch5x5 = 32
        pool_proj = 32
        # Modelo para volúmenes 3D (pequeño)
        self.model_3d_small = nn.Sequential(
            nn.Conv3d(in_channels_3d, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            InceptionBlock3D(64, ch1x1, 48, ch3x3, 16, ch5x5, pool_proj),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(ch1x1 + ch3x3 + ch5x5 + pool_proj, 128)
        )

        self.model_3d_large = nn.Sequential(
            nn.Conv3d(in_channels_3d, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            InceptionBlock3D(128, ch1x1, 48, ch3x3, 16, ch5x5, pool_proj),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(ch1x1 + ch3x3 + ch5x5 + pool_proj, 128)
        )


        self.fc_descriptors = nn.Sequential(
            nn.Linear(num_descriptors, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),  # Capa añadida para igualar DenseNet
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.Dropout(0.3)  # Dropout para evitar sobreajuste
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(128 + 128 + 64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128)  # Capa adicional para igualar DenseNet
        )

        self.output_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),  # Capa adicional para igualar DenseNet
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, image=None, volume_small=None, volume_large=None, descriptors=None):
        features = []
        
        if image is not None:
            x_2d = self.model_2d(image)
            #print("Salida de model_2d:", x_2d.shape)
            features.append(x_2d)
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
