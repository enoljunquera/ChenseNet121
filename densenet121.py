import torch
import torch.nn as nn
import torch.nn.functional as F

# Bloque Denso: cada capa usa la salida de todas las anteriores
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers, is_3d):
        super(DenseBlock, self).__init__()
        conv_layer = nn.Conv3d if is_3d else nn.Conv2d
        batch_norm_layer = nn.BatchNorm3d if is_3d else nn.BatchNorm2d
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                batch_norm_layer(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                conv_layer(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
            )
            for i in range(num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        return x
        
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, is_3d):
        super(TransitionLayer, self).__init__()
        conv_layer = nn.Conv3d if is_3d else nn.Conv2d
        batch_norm_layer = nn.BatchNorm3d if is_3d else nn.BatchNorm2d
        pool_layer = nn.AvgPool3d if is_3d else nn.AvgPool2d
        
        self.batch_norm = batch_norm_layer(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = conv_layer(in_channels, out_channels, kernel_size=1, bias=False)
        
        # Pooling solo si la dimensión espacial es mayor a 2
        self.pool = pool_layer(kernel_size=2, stride=2) if is_3d else pool_layer(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv(x)

        # Verifica si el tamaño espacial es mayor a 1 antes de aplicar pooling
        if x.shape[-1] > 1:  
            x = self.pool(x)

        return x


# Modelo principal
class FlexibleDenseNet(nn.Module):
    def __init__(self, growth_rate, num_classes):
        super(FlexibleDenseNet, self).__init__()
        
        self.model_2d = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Bloque denso 1
            DenseBlock(64, growth_rate, num_layers=6, is_3d=False),
            TransitionLayer(64 + 6 * growth_rate, 128, is_3d=False),

            # Bloque denso 2
            DenseBlock(128, growth_rate, num_layers=12, is_3d=False),
            TransitionLayer(128 + 12 * growth_rate, 256, is_3d=False),

            # Bloque denso 3
            DenseBlock(256, growth_rate, num_layers=24, is_3d=False),
            TransitionLayer(256 + 24 * growth_rate, 512, is_3d=False),

            # Bloque denso 4
            DenseBlock(512, growth_rate, num_layers=16, is_3d=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            
            # Cálculo dinámico del número de canales para la capa lineal
            nn.Linear(512 + 16 * growth_rate, 512)
        )

        self.model_3d_small = nn.Sequential(
            DenseBlock(8, growth_rate, num_layers=4, is_3d=True),
            TransitionLayer(8 + 4 * growth_rate, 32, is_3d=True),  # Aumentamos a 32 en lugar de 64

            DenseBlock(32, growth_rate, num_layers=8, is_3d=True),
            TransitionLayer(32 + 8 * growth_rate, 64, is_3d=True),

            DenseBlock(64, growth_rate, num_layers=12, is_3d=True),
            TransitionLayer(64 + 12 * growth_rate, 128, is_3d=True),

            DenseBlock(128, growth_rate, num_layers=16, is_3d=True),
            
            # Ajuste de canales antes de la convolución final
            nn.Conv3d(128 + 16 * growth_rate, 8, kernel_size=1, bias=False),  
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(8),

            # Conv 3x3 para aumentar a 32 canales
            nn.Conv3d(8, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(32),

            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(32, 64)
        )

        self.model_3d_large = nn.Sequential(
            DenseBlock(8, growth_rate, num_layers=4, is_3d=True),
            TransitionLayer(8 + 4 * growth_rate, 64, is_3d=True),

            DenseBlock(64, growth_rate, num_layers=8, is_3d=True),
            TransitionLayer(64 + 8 * growth_rate, 128, is_3d=True),

            DenseBlock(128, growth_rate, num_layers=12, is_3d=True),
            TransitionLayer(128 + 12 * growth_rate, 256, is_3d=True),

            DenseBlock(256, growth_rate, num_layers=16, is_3d=True),
            TransitionLayer(256 + 16 * growth_rate, 512, is_3d=True),

            # Ajuste de canales antes de la convolución final
            nn.Conv3d(512, 8, kernel_size=1, bias=False),  
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(8),

            nn.Conv3d(8, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(64),

            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(64, 128)
        )

        
        # Procesamiento de descriptores flotantes (4 iniciales + 2 adicionales)
        self.fc_descriptors = nn.Sequential(
            nn.Linear(6, 128),  # Aumentamos la dimensión de salida
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),  # Reducimos a 64 para mantener compatibilidad con fusion_layer
            nn.ReLU(),
            nn.Dropout(0.3)  # Dropout para evitar sobreajuste
        )

        # Capa de fusión intermedia antes de la final
        self.fusion_layer = nn.Sequential(
            nn.Linear( 128 + 64 + 512, 256),  # En vez de ir directo a 128, pasamos por 256
            nn.ReLU(),
            nn.Linear(256, 128),  # Luego bajamos a 128
            
            
            nn.ReLU(inplace=True),
            nn.Linear(128, 128)  # Capa adicional para igualar DenseNet
        )

        # Capa de salida
        self.output_layer = nn.Sequential(
            nn.Linear(128, 64),
            
            
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),  # Capa adicional para igualar DenseNet
            

            nn.ReLU(),   
            nn.Linear(64, 1)  # Predicción final como un flotante
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
