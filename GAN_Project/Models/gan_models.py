import torch
import torch.nn as nn
import torch.nn.functional as F
# Import the device from config
from configs.config import Config
config = Config()

class ConditionalGenerator(nn.Module):
    def __init__(self, config):
        super(ConditionalGenerator, self).__init__()
        
        # Encoder for damaged input
        self.encoder = nn.Sequential(
            # Initial layer - no batch norm
            nn.Conv2d(config.CHANNELS, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Downsampling layers
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder for reconstruction
        self.decoder = nn.ModuleList([
            # First upsampling layer
            nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Second upsampling layer - takes 512+512 channels from skip connection
            nn.ConvTranspose2d(1024, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Third upsampling layer - takes 256+256 channels from skip connection
            nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Fourth upsampling layer - takes 128+128 channels from skip connection
            nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Final layer - takes 64+64 channels from skip connection
            nn.ConvTranspose2d(128, config.CHANNELS, 4, stride=2, padding=1),
            nn.Tanh()
        ])
        
        # Skip connection layers
        self.skip_connection1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.skip_connection2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.skip_connection3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.skip_connection4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Store encoder outputs for skip connections
        encoder_outputs = []
        
        # First layer
        e1 = self.encoder[0:2](x)
        encoder_outputs.append(e1)
        
        # Second layer
        e2 = self.encoder[2:5](e1)
        encoder_outputs.append(e2)
        
        # Third layer
        e3 = self.encoder[5:8](e2)
        encoder_outputs.append(e3)
        
        # Fourth layer
        e4 = self.encoder[8:11](e3)
        encoder_outputs.append(e4)
        
        # Fifth layer (bottleneck)
        bottleneck = self.encoder[11:14](e4)
        
        # Start decoding with first upsampling layer
        d1 = self.decoder[0](bottleneck)
        d1 = self.decoder[1](d1)
        d1 = self.decoder[2](d1)
        
        # Apply first skip connection and second upsampling layer
        skip4 = self.skip_connection4(encoder_outputs[3])
        d1_skip = torch.cat([d1, skip4], dim=1)
        d2 = self.decoder[3](d1_skip)
        d2 = self.decoder[4](d2)
        d2 = self.decoder[5](d2)
        
        # Apply second skip connection and third upsampling layer
        skip3 = self.skip_connection3(encoder_outputs[2])
        d2_skip = torch.cat([d2, skip3], dim=1)
        d3 = self.decoder[6](d2_skip)
        d3 = self.decoder[7](d3)
        d3 = self.decoder[8](d3)
        
        # Apply third skip connection and fourth upsampling layer
        skip2 = self.skip_connection2(encoder_outputs[1])
        d3_skip = torch.cat([d3, skip2], dim=1)
        d4 = self.decoder[9](d3_skip)
        d4 = self.decoder[10](d4)
        d4 = self.decoder[11](d4)
        
        # Apply fourth skip connection and final layer
        skip1 = self.skip_connection1(encoder_outputs[0])
        d4_skip = torch.cat([d4, skip1], dim=1)
        output = self.decoder[12](d4_skip)
        output = self.decoder[13](output)
        
        return output

class ConditionalDiscriminator(nn.Module):
    def __init__(self, config):
        super(ConditionalDiscriminator, self).__init__()
        
        # Combined input: damaged image + either real or generated complete image
        # This enforces the conditional aspect of the GAN
        self.model = nn.Sequential(
            # Layer 1
            nn.Conv2d(config.CHANNELS * 2, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 5 - output layer (no sigmoid - we'll use BCEWithLogitsLoss)
            nn.Conv2d(512, 1, 4, stride=1, padding=1)
        )
    
    def forward(self, condition, target):
        # Combine the condition (damaged image) with the target (complete image)
        combined = torch.cat([condition, target], dim=1)
        return self.model(combined)
