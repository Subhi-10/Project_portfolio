import torch
import torch.nn as nn
import torch.nn.functional as F
# Import the device from config
from configs.config import Config
config = Config()

class GatedConv2d(nn.Module):
    """ Gated Convolution Layer for selective feature processing """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=nn.ReLU()):
        super(GatedConv2d, self).__init__()
        self.activation = activation
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_mask = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.conv_feature(x)
        mask = self.sigmoid(self.conv_mask(x))
        return self.activation(features * mask)

class GatedGenerator(nn.Module):
    """ Generator using Gated Convolutions with Edge-Connect style architecture """
    def __init__(self, config):
        super(GatedGenerator, self).__init__()
        
        # Encoder Network
        self.enc_conv1 = GatedConv2d(config.CHANNELS, 64, 7, stride=1, padding=3)
        self.enc_conv2 = GatedConv2d(64, 128, 4, stride=2, padding=1)
        self.enc_conv3 = GatedConv2d(128, 256, 4, stride=2, padding=1)
        self.enc_conv4 = GatedConv2d(256, 512, 4, stride=2, padding=1)
        
        # Middle Network
        self.middle_conv = GatedConv2d(512, 512, 3, padding=1)
        
        # Refinement Network
        self.refine_conv1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.refine_bn1 = nn.BatchNorm2d(256)
        
        self.refine_conv2 = nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1)
        self.refine_bn2 = nn.BatchNorm2d(128)
        
        self.refine_conv3 = nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1)
        self.refine_bn3 = nn.BatchNorm2d(64)
        
        self.refine_conv4 = nn.Conv2d(128, config.CHANNELS, 7, stride=1, padding=3)
        
        # Activation
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # Encoder
        e1 = self.enc_conv1(x)
        e2 = self.enc_conv2(e1)
        e3 = self.enc_conv3(e2)
        e4 = self.enc_conv4(e3)
        
        # Middle
        m = self.middle_conv(e4)
        
        # Refinement with skip connections
        r1 = self.relu(self.refine_bn1(self.refine_conv1(m)))
        r1_cat = torch.cat([r1, e3], dim=1)
        r2 = self.relu(self.refine_bn2(self.refine_conv2(r1_cat)))
        r2_cat = torch.cat([r2, e2], dim=1)
        r3 = self.relu(self.refine_bn3(self.refine_conv3(r2_cat)))
        r3_cat = torch.cat([r3, e1], dim=1)
        output = self.tanh(self.refine_conv4(r3_cat))
        
        return output
    
class GatedEdgeGenerator(nn.Module):
    """ Enhanced Generator with both edge detection and completion paths """
    def __init__(self, config):
        super(GatedEdgeGenerator, self).__init__()
        
        # Edge Detection Network
        self.edge_conv1 = GatedConv2d(config.CHANNELS, 64, 7, padding=3)
        self.edge_conv2 = GatedConv2d(64, 128, 4, stride=2, padding=1)
        self.edge_conv3 = GatedConv2d(128, 256, 4, stride=2, padding=1)
        self.edge_conv4 = GatedConv2d(256, 512, 4, stride=2, padding=1)
        
        # Edge Refinement Network
        self.edge_refine_conv1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.edge_refine_bn1 = nn.BatchNorm2d(256)
        
        self.edge_refine_conv2 = nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1)
        self.edge_refine_bn2 = nn.BatchNorm2d(128)
        
        self.edge_refine_conv3 = nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1)
        self.edge_refine_bn3 = nn.BatchNorm2d(64)
        
        self.edge_refine_conv4 = nn.Conv2d(128, 1, 7, padding=3)
        
        # Completion Network (with gated convolutions)
        self.complete_conv1 = GatedConv2d(config.CHANNELS + 1, 64, 7, padding=3)
        self.complete_conv2 = GatedConv2d(64, 128, 4, stride=2, padding=1)
        self.complete_conv3 = GatedConv2d(128, 256, 4, stride=2, padding=1)
        self.complete_conv4 = GatedConv2d(256, 512, 4, stride=2, padding=1)
        
        # Completion Decoder with skip connections
        self.complete_deconv1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.complete_bn1 = nn.BatchNorm2d(256)
        
        self.complete_deconv2 = nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1)
        self.complete_bn2 = nn.BatchNorm2d(128)
        
        self.complete_deconv3 = nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1)
        self.complete_bn3 = nn.BatchNorm2d(64)
        
        self.complete_deconv4 = nn.Conv2d(128, config.CHANNELS, 7, padding=3)
        
        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # Edge Detection
        e1 = self.edge_conv1(x)
        e2 = self.edge_conv2(e1)
        e3 = self.edge_conv3(e2)
        e4 = self.edge_conv4(e3)
        
        # Edge Refinement
        r1 = self.relu(self.edge_refine_bn1(self.edge_refine_conv1(e4)))
        r1_cat = torch.cat([r1, e3], dim=1)
        r2 = self.relu(self.edge_refine_bn2(self.edge_refine_conv2(r1_cat)))
        r2_cat = torch.cat([r2, e2], dim=1)
        r3 = self.relu(self.edge_refine_bn3(self.edge_refine_conv3(r2_cat)))
        r3_cat = torch.cat([r3, e1], dim=1)
        edge_map = self.tanh(self.edge_refine_conv4(r3_cat))
        
        # Completion Network with edge map as input
        x_edge = torch.cat([x, edge_map], dim=1)
        c1 = self.complete_conv1(x_edge)
        c2 = self.complete_conv2(c1)
        c3 = self.complete_conv3(c2)
        c4 = self.complete_conv4(c3)
        
        # Completion Decoder
        d1 = self.relu(self.complete_bn1(self.complete_deconv1(c4)))
        d1_cat = torch.cat([d1, c3], dim=1)
        d2 = self.relu(self.complete_bn2(self.complete_deconv2(d1_cat)))
        d2_cat = torch.cat([d2, c2], dim=1)
        d3 = self.relu(self.complete_bn3(self.complete_deconv3(d2_cat)))
        d3_cat = torch.cat([d3, c1], dim=1)
        completed = self.tanh(self.complete_deconv4(d3_cat))
        
        return edge_map, completed
    
class GatedDiscriminator(nn.Module):
    """ Discriminator for Gated GAN with conditional input """
    def __init__(self, config):
        super(GatedDiscriminator, self).__init__()
        
        def disc_block(in_channels, out_channels, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *disc_block(config.CHANNELS * 2, 64, normalize=False),
            *disc_block(64, 128),
            *disc_block(128, 256),
            *disc_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, real_img, damaged_img):
        # Concatenate real and damaged images along channel dimension
        combined = torch.cat([real_img, damaged_img], dim=1)
        return self.model(combined)

class GatedEdgeDiscriminator(nn.Module):
    """ Enhanced Discriminator that also takes edge map into account """
    def __init__(self, config):
        super(GatedEdgeDiscriminator, self).__init__()
        
        def disc_block(in_channels, out_channels, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *disc_block(config.CHANNELS * 2 + 1, 64, normalize=False),  # +1 for edge map
            *disc_block(64, 128),
            *disc_block(128, 256),
            *disc_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, img, edge_map, gen_img):
        # Concatenate real image, edge map and generated image
        combined = torch.cat([img, edge_map, gen_img], dim=1)
        return self.model(combined)
