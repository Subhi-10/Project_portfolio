import torch
import torch.nn as nn
import torch.nn.functional as F
# Import the device from config
from configs.config import Config
config = Config()

class EdgeGenerator(nn.Module):
    def __init__(self, config):
        super(EdgeGenerator, self).__init__()
        
        # Edge Detection Network
        self.edge_conv1 = nn.Conv2d(config.CHANNELS, 64, 7, stride=1, padding=3)
        self.edge_conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.edge_conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.edge_conv4 = nn.Conv2d(256, 512, 4, stride=2, padding=1)
        
        # Edge Refinement Network
        self.refine_conv1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.refine_conv2 = nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1)
        self.refine_conv3 = nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1)
        self.refine_conv4 = nn.Conv2d(128, 1, 7, stride=1, padding=3)
        
        # Completion Network
        self.complete_conv1 = nn.Conv2d(config.CHANNELS + 1, 64, 7, stride=1, padding=3)
        self.complete_conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.complete_conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.complete_conv4 = nn.Conv2d(256, 512, 4, stride=2, padding=1)
        
        # Completion Decoder
        self.complete_deconv1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.complete_deconv2 = nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1)
        self.complete_deconv3 = nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1)
        self.complete_deconv4 = nn.Conv2d(128, config.CHANNELS, 7, stride=1, padding=3)
        
        # Batch normalization layers
        self.bn_e2 = nn.BatchNorm2d(128)
        self.bn_e3 = nn.BatchNorm2d(256)
        self.bn_e4 = nn.BatchNorm2d(512)
        
        self.bn_r1 = nn.BatchNorm2d(256)
        self.bn_r2 = nn.BatchNorm2d(128)
        self.bn_r3 = nn.BatchNorm2d(64)
        
        self.bn_c2 = nn.BatchNorm2d(128)
        self.bn_c3 = nn.BatchNorm2d(256)
        self.bn_c4 = nn.BatchNorm2d(512)
        
        self.bn_d1 = nn.BatchNorm2d(256)
        self.bn_d2 = nn.BatchNorm2d(128)
        self.bn_d3 = nn.BatchNorm2d(64)
        
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # Edge Detection
        e1 = self.relu(self.edge_conv1(x))
        e2 = self.relu(self.bn_e2(self.edge_conv2(e1)))
        e3 = self.relu(self.bn_e3(self.edge_conv3(e2)))
        e4 = self.relu(self.bn_e4(self.edge_conv4(e3)))
        
        # Edge Refinement
        r1 = self.relu(self.bn_r1(self.refine_conv1(e4)))
        r1_cat = torch.cat([r1, e3], dim=1)
        r2 = self.relu(self.bn_r2(self.refine_conv2(r1_cat)))
        r2_cat = torch.cat([r2, e2], dim=1)
        r3 = self.relu(self.bn_r3(self.refine_conv3(r2_cat)))
        r3_cat = torch.cat([r3, e1], dim=1)
        edge_map = self.tanh(self.refine_conv4(r3_cat))
        
        # Completion Network
        x_edge = torch.cat([x, edge_map], dim=1)
        c1 = self.relu(self.complete_conv1(x_edge))
        c2 = self.relu(self.bn_c2(self.complete_conv2(c1)))
        c3 = self.relu(self.bn_c3(self.complete_conv3(c2)))
        c4 = self.relu(self.bn_c4(self.complete_conv4(c3)))
        
        # Completion Decoder
        d1 = self.relu(self.bn_d1(self.complete_deconv1(c4)))
        d1_cat = torch.cat([d1, c3], dim=1)
        d2 = self.relu(self.bn_d2(self.complete_deconv2(d1_cat)))
        d2_cat = torch.cat([d2, c2], dim=1)
        d3 = self.relu(self.bn_d3(self.complete_deconv3(d2_cat)))
        d3_cat = torch.cat([d3, c1], dim=1)
        completed = self.tanh(self.complete_deconv4(d3_cat))
        
        return edge_map, completed
    
class EdgeDiscriminator(nn.Module):
    def __init__(self, config):
        super(EdgeDiscriminator, self).__init__()
        
        def discriminator_block(in_channels, out_channels, normalization=True):
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2))
            return layers
        
        self.model = nn.Sequential(
            *discriminator_block(config.CHANNELS * 2 + 1, 64, False),  # +1 for edge map
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, img, edge_map, gen_img):
        combined = torch.cat([img, edge_map, gen_img], dim=1)
        return self.model(combined)
