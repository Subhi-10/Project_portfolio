import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# Import the device from config
from configs.config import Config
config = Config()

class AdaIN(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.norm = nn.InstanceNorm2d(n_channels)
        self.style_scale = nn.Linear(512, n_channels)
        self.style_bias = nn.Linear(512, n_channels)
        
    def forward(self, x, w):
        normalized = self.norm(x)
        scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        return normalized * scale + bias

class SynthesisLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.adain = AdaIN(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x, w):
        x = self.conv(x)
        x = self.adain(x, w)
        return self.activation(x)

class StyleGAN3Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Reduce the complexity of the mapping network
        self.mapping = nn.Sequential(
            nn.Linear(config.CHANNELS * 32, 512),  # Reduced input size
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2)
        )
        
        # Initial input processing
        self.input_process = nn.Sequential(
            nn.Conv2d(config.CHANNELS, 32, 3, padding=1),  # Reduced to 32 channels
            nn.LeakyReLU(0.2)
        )
        
        # Main synthesis network - reduced channel sizes
        self.synthesis_blocks = nn.ModuleList([
            SynthesisLayer(32, 64),
            SynthesisLayer(64, 128),
            SynthesisLayer(128, 128),  # Reduced from 256
            SynthesisLayer(128, 64),   # Reduced from 256, 128
            SynthesisLayer(64, 32)     # Reduced from 128, 64
        ])
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Conv2d(32, config.CHANNELS, 3, padding=1),
            nn.Tanh()
        )
        
        # Fourier feature mapping
        self.register_buffer('fourier_freqs', torch.randn(16, 2) * 10.)  # Reduced from 32 to 16
        
    def fourier_features(self, coords):
        f = 2 * math.pi * coords @ self.fourier_freqs.T
        return torch.cat([torch.sin(f), torch.cos(f)], dim=-1)
    
    def forward(self, x):
        # Generate style vector - using average pooling to reduce dimensions
        batch_size = x.size(0)
        pooled_x = F.adaptive_avg_pool2d(x, (32, 1))  # Reduce spatial dimensions
        w = self.mapping(pooled_x.view(batch_size, -1))
        
        # Initial processing
        h = self.input_process(x)
        
        # Apply synthesis blocks with style modulation
        for block in self.synthesis_blocks:
            h = block(h, w)
        
        # Generate output image
        output = self.output_layer(h)
        
        return output

class StyleGAN3Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        def discriminator_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.GroupNorm(4, out_channels),  # Reduced from 8 to 4 groups
                nn.LeakyReLU(0.2)
            )
        
        self.main = nn.Sequential(
            # Input layer
            nn.Conv2d(config.CHANNELS * 2, 32, 4, stride=2, padding=1),  # Reduced from 64 to 32
            nn.LeakyReLU(0.2),
            
            # Downsampling blocks - reduced channel sizes
            discriminator_block(32, 64),   # Reduced from 64, 128
            discriminator_block(64, 128),  # Reduced from 128, 256
            discriminator_block(128, 256), # Reduced from 256, 512
            
            # Output layer
            nn.Conv2d(256, 1, 4, stride=1, padding=0),
            nn.Sigmoid()
        )
        
        # Fourier feature mapping - reduced size
        self.register_buffer('fourier_freqs', torch.randn(16, 2) * 10.)  # Reduced from 32 to 16
    
    def forward(self, damaged, complete):
        # Concatenate input images
        x = torch.cat([damaged, complete], dim=1)
        return self.main(x)
