import torch
import torch.nn as nn
import torch.nn.functional as F
# Import the device from config
from configs.config import Config
config = Config()

def safe_checkpoint(function, *args):
    """Wrapper for torch.utils.checkpoint to handle parameter compatibility issues"""
    try:
        # Try with use_reentrant=False first (recommended)
        return torch.utils.checkpoint.checkpoint(function, *args, use_reentrant=False)
    except TypeError:
        try:
            # Fall back to use_reentrant=True if False doesn't work
            return torch.utils.checkpoint.checkpoint(function, *args, use_reentrant=True)
        except TypeError:
            # Finally, try without the parameter if both fail
            return torch.utils.checkpoint.checkpoint(function, *args)

class GatedConv2d(nn.Module):
    """Gated Convolution layer with improved feature processing"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, activation=nn.LeakyReLU(0.2)):
        super(GatedConv2d, self).__init__()
        self.activation = activation
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.conv_mask = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.sigmoid = nn.Sigmoid()
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        features = self.conv_feature(x)
        features = self.batch_norm(features)
        mask = self.sigmoid(self.conv_mask(x))
        return self.activation(features * mask)

class GatedDeconv2d(nn.Module):
    """Enhanced Gated Deconvolution layer for better upsampling"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=nn.LeakyReLU(0.2)):
        super(GatedDeconv2d, self).__init__()
        self.activation = activation
        self.deconv_feature = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.deconv_mask = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.sigmoid = nn.Sigmoid()
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        features = self.deconv_feature(x)
        features = self.batch_norm(features)
        mask = self.sigmoid(self.deconv_mask(x))
        return self.activation(features * mask)

class AttentionModule(nn.Module):
    """Attention module to focus on relevant features with memory optimization"""
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        # Reduce channel dimensionality for more efficient attention computation
        self.conv1 = nn.Conv2d(in_channels, max(in_channels // 16, 8), kernel_size=1)
        self.conv2 = nn.Conv2d(max(in_channels // 16, 8), in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Project to lower dimension more aggressively
        proj_query = self.conv1(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        proj_key = self.conv1(x).view(batch_size, -1, height * width)
        
        # Calculate attention map
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        
        # Apply attention
        proj_value = x.view(batch_size, channels, -1)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Residual connection
        return x + out * 0.1  # Scale down the attention effect to save memory

class EdgeGenerator(nn.Module):
    """Memory-optimized Edge prediction network"""
    def __init__(self, config):
        super(EdgeGenerator, self).__init__()
        
        # Encoder with reduced filter sizes
        self.enc_conv1 = GatedConv2d(config.CHANNELS, 32, 7, stride=1, padding=3)
        self.enc_conv2 = GatedConv2d(32, 64, 4, stride=2, padding=1)
        self.enc_conv3 = GatedConv2d(64, 128, 4, stride=2, padding=1)
        self.enc_conv4 = GatedConv2d(128, 256, 4, stride=2, padding=1)
        
        # Add attention module
        self.attention = AttentionModule(256)
        
        # Dilated convolutions with reduced parameters
        self.dil_conv1 = GatedConv2d(256, 256, 3, padding=2, dilation=2)
        self.dil_conv2 = GatedConv2d(256, 256, 3, padding=4, dilation=4)
        
        # Decoder with skip connections
        self.dec_conv1 = GatedDeconv2d(256, 128, 4, stride=2, padding=1)
        self.dec_conv2 = GatedDeconv2d(256, 64, 4, stride=2, padding=1)
        self.dec_conv3 = GatedDeconv2d(128, 32, 4, stride=2, padding=1)
        self.dec_conv4 = nn.Conv2d(64, 1, 7, stride=1, padding=3)
        
        self.relu = nn.LeakyReLU(0.2, True)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # Encoder
        e1 = self.enc_conv1(x)
        e2 = self.enc_conv2(e1)
        e3 = self.enc_conv3(e2)
        e4 = self.enc_conv4(e3)
        
        # Apply attention
        e4 = self.attention(e4)
        
        # Dilated convolutions (reduced number)
        d = self.dil_conv1(e4)
        d = self.dil_conv2(d)
        
        # Decoder with skip connections
        # Use gradient checkpointing to save memory
        if self.training and torch.cuda.is_available():
            r1 = safe_checkpoint(self.dec_conv1, d)
            r1_cat = torch.cat([r1, e3], dim=1)
            r2 = safe_checkpoint(self.dec_conv2, r1_cat)
            r2_cat = torch.cat([r2, e2], dim=1)
            r3 = safe_checkpoint(self.dec_conv3, r2_cat)
            r3_cat = torch.cat([r3, e1], dim=1)
            edge_map = self.tanh(self.dec_conv4(r3_cat))
        else:
            r1 = self.dec_conv1(d)
            r1_cat = torch.cat([r1, e3], dim=1)
            r2 = self.dec_conv2(r1_cat)
            r2_cat = torch.cat([r2, e2], dim=1)
            r3 = self.dec_conv3(r2_cat)
            r3_cat = torch.cat([r3, e1], dim=1)
            edge_map = self.tanh(self.dec_conv4(r3_cat))
        
        return edge_map

class CompletionGenerator(nn.Module):
    """Memory-optimized texture completion network"""
    def __init__(self, config):
        super(CompletionGenerator, self).__init__()
        
        # Encoder with edge input and reduced parameters
        self.enc_conv1 = GatedConv2d(config.CHANNELS + 1, 32, 7, stride=1, padding=3)
        self.enc_conv2 = GatedConv2d(32, 64, 4, stride=2, padding=1)
        self.enc_conv3 = GatedConv2d(64, 128, 4, stride=2, padding=1)
        self.enc_conv4 = GatedConv2d(128, 256, 4, stride=2, padding=1)
        
        # Add attention module
        self.attention = AttentionModule(256)
        
        # Dilated convolutions with reduced parameters
        self.dil_conv1 = GatedConv2d(256, 256, 3, padding=2, dilation=2)
        self.dil_conv2 = GatedConv2d(256, 256, 3, padding=4, dilation=4)
        
        # Decoder with skip connections
        self.dec_conv1 = GatedDeconv2d(256, 128, 4, stride=2, padding=1)
        self.dec_conv2 = GatedDeconv2d(256, 64, 4, stride=2, padding=1)
        self.dec_conv3 = GatedDeconv2d(128, 32, 4, stride=2, padding=1)
        self.dec_conv4 = nn.Conv2d(64, config.CHANNELS, 7, stride=1, padding=3)
        
        self.relu = nn.LeakyReLU(0.2, True)
        self.tanh = nn.Tanh()
        
    def forward(self, x, edge_map):
        # Combine input with edge map
        x_edge = torch.cat([x, edge_map], dim=1)
        
        # Encoder
        e1 = self.enc_conv1(x_edge)
        e2 = self.enc_conv2(e1)
        e3 = self.enc_conv3(e2)
        e4 = self.enc_conv4(e3)
        
        # Apply attention
        e4 = self.attention(e4)
        
        # Dilated convolutions (reduced)
        d = self.dil_conv1(e4)
        d = self.dil_conv2(d)
        
        # Decoder with skip connections
        # Use gradient checkpointing to save memory
        if self.training and torch.cuda.is_available():
            r1 = safe_checkpoint(self.dec_conv1, d)
            r1_cat = torch.cat([r1, e3], dim=1)
            r2 = safe_checkpoint(self.dec_conv2, r1_cat)
            r2_cat = torch.cat([r2, e2], dim=1)
            r3 = safe_checkpoint(self.dec_conv3, r2_cat)
            r3_cat = torch.cat([r3, e1], dim=1)
            completed = self.tanh(self.dec_conv4(r3_cat))
        else:
            r1 = self.dec_conv1(d)
            r1_cat = torch.cat([r1, e3], dim=1)
            r2 = self.dec_conv2(r1_cat)
            r2_cat = torch.cat([r2, e2], dim=1)
            r3 = self.dec_conv3(r2_cat)
            r3_cat = torch.cat([r3, e1], dim=1)
            completed = self.tanh(self.dec_conv4(r3_cat))
        
        return completed

class BackgroundDetector(nn.Module):
    """Module to distinguish between background and artifact"""
    def __init__(self, config):
        super(BackgroundDetector, self).__init__()
        self.conv1 = nn.Conv2d(config.CHANNELS, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return self.sigmoid(self.conv3(x))

class HybridGenerator(nn.Module):
    """Combined edge prediction and texture completion generator with memory optimization"""
    def __init__(self, config):
        super(HybridGenerator, self).__init__()
        self.edge_generator = EdgeGenerator(config)
        self.completion_generator = CompletionGenerator(config)
        self.background_detector = BackgroundDetector(config)
        
    def forward(self, x):
        # Generate edge map
        edge_map = self.edge_generator(x)
        
        # Generate background mask
        bg_mask = self.background_detector(x)
        
        # Generate completed image
        completed = self.completion_generator(x, edge_map)
        
        # Apply background mask
        masked_completion = x * bg_mask + completed * (1 - bg_mask)
        
        return edge_map, masked_completion

class HybridDiscriminator(nn.Module):
    """Memory-optimized discriminator"""
    def __init__(self, config):
        super(HybridDiscriminator, self).__init__()
        
        # Edge discriminator with reduced parameters
        self.edge_discriminator = nn.Sequential(
            nn.Conv2d(1 + config.CHANNELS, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 4, padding=1),
            nn.Sigmoid()
        )
        
        # Texture discriminator with reduced parameters
        self.texture_discriminator = nn.Sequential(
            GatedConv2d(config.CHANNELS * 2 + 1, 32, 4, stride=2, padding=1, activation=nn.LeakyReLU(0.2, inplace=True)),
            GatedConv2d(32, 64, 4, stride=2, padding=1, activation=nn.LeakyReLU(0.2, inplace=True)),
            GatedConv2d(64, 128, 4, stride=2, padding=1, activation=nn.LeakyReLU(0.2, inplace=True)),
            AttentionModule(128),
            nn.Conv2d(128, 1, 4, padding=1),
            nn.Sigmoid()
        )
        
        # Background-artifact separation discriminator with reduced parameters
        self.background_discriminator = nn.Sequential(
            GatedConv2d(config.CHANNELS * 2, 32, 4, stride=2, padding=1, activation=nn.LeakyReLU(0.2, inplace=True)),
            GatedConv2d(32, 64, 4, stride=2, padding=1, activation=nn.LeakyReLU(0.2, inplace=True)),
            nn.Conv2d(64, 1, 4, padding=1),
            nn.Sigmoid()
        )
    
    def forward_edge(self, edge_map, damaged_img):
        # Combine edge map with damaged image
        combined = torch.cat([edge_map, damaged_img], dim=1)
        return self.edge_discriminator(combined)
    
    def forward_texture(self, completed_img, edge_map, damaged_img):
        # Combine completed image, edge map and damaged image
        combined = torch.cat([completed_img, edge_map, damaged_img], dim=1)
        return self.texture_discriminator(combined)
    
    def forward_background(self, completed_img, damaged_img):
        # Evaluate background-artifact separation
        combined = torch.cat([completed_img, damaged_img], dim=1)
        return self.background_discriminator(combined)
    
    def forward(self, edge_map, completed_img, damaged_img):
        # Get all discrimination results
        edge_validity = self.forward_edge(edge_map, damaged_img)
        texture_validity = self.forward_texture(completed_img, edge_map, damaged_img)
        bg_validity = self.forward_background(completed_img, damaged_img)
        return edge_validity, texture_validity, bg_validity
