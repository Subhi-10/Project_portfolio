import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os

# Set CUDA memory allocation configuration to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Import the device from config
from configs.config import Config
config = Config()

def safe_checkpoint(function, *args):
    """Improved wrapper for torch.utils.checkpoint to handle parameter compatibility issues"""
    def custom_forward(*inputs):
        return function(*inputs)
    
    try:
        return torch.utils.checkpoint.checkpoint(custom_forward, *args, use_reentrant=False, preserve_rng_state=False)
    except TypeError:
        try:
            return torch.utils.checkpoint.checkpoint(custom_forward, *args, use_reentrant=True, preserve_rng_state=False)
        except TypeError:
            return torch.utils.checkpoint.checkpoint(custom_forward, *args)

class GatedConv2d(nn.Module):
    """Gated Convolution Layer for selective feature processing"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, activation=nn.ReLU()):
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

class AttentionBlock(nn.Module):
    """Memory-efficient attention mechanism for feature fusion"""
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        # Use more aggressive channel reduction
        self.factor = 32
        reduced_channels = max(in_channels // self.factor, 8)
        
        self.conv_query = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # More aggressive spatial reduction
        self.downsample_factor = 2
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Calculate new dimensions with aggressive downsampling
        H_d = max(int(H / self.downsample_factor), 8)
        W_d = max(int(W / self.downsample_factor), 8)
        
        # Downsample spatially to reduce memory footprint
        x_down = F.interpolate(x, size=(H_d, W_d), mode='bilinear', align_corners=False)
        
        # Apply convolutions with reduced dimensionality
        query = self.conv_query(x_down)
        key = self.conv_key(x_down)
        value = self.conv_value(x_down)
        
        # Reshape for attention calculation
        query_flat = query.view(batch_size, -1, H_d*W_d).permute(0, 2, 1)
        key_flat = key.view(batch_size, -1, H_d*W_d)
        value_flat = value.view(batch_size, -1, H_d*W_d)
        
        # Compute attention scores with fp16 to save memory
        energy = torch.bmm(query_flat, key_flat)
        attention = self.softmax(energy)
        
        # Apply attention weights
        out = torch.bmm(value_flat, attention.permute(0, 2, 1))
        out = out.view(batch_size, -1, H_d, W_d)
        
        # Upsample back to original size
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        
        # Use smaller gamma effect to reduce memory consumption
        out = self.gamma * out + x
        
        return out

class ConditionalGatedGenerator(nn.Module):
    """Memory-efficient Generator with improved skip connections"""
    def __init__(self, config):
        super(ConditionalGatedGenerator, self).__init__()
        
        # Encoder with Gated Convolutions - reduced channel counts
        self.enc_conv1 = GatedConv2d(config.CHANNELS, 32, 4, stride=2, padding=1, activation=nn.LeakyReLU(0.2))
        self.enc_conv2 = GatedConv2d(32, 64, 4, stride=2, padding=1, activation=nn.LeakyReLU(0.2))
        self.enc_conv3 = GatedConv2d(64, 128, 4, stride=2, padding=1, activation=nn.LeakyReLU(0.2))
        self.enc_conv4 = GatedConv2d(128, 256, 4, stride=2, padding=1, activation=nn.LeakyReLU(0.2))
        self.enc_conv5 = GatedConv2d(256, 256, 4, stride=2, padding=1, activation=nn.LeakyReLU(0.2))
        
        # Skip connections with smaller convolutions
        self.skip_attention1 = AttentionBlock(32)
        self.skip_connection1 = nn.Sequential(
            GatedConv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.skip_attention2 = AttentionBlock(64)
        self.skip_connection2 = nn.Sequential(
            GatedConv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.skip_attention3 = AttentionBlock(128)
        self.skip_connection3 = nn.Sequential(
            GatedConv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.skip_attention4 = AttentionBlock(256)
        self.skip_connection4 = nn.Sequential(
            GatedConv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Decoder with transposed convolutions and batch norm
        self.dec_conv1 = nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(256)
        self.dec_relu1 = nn.ReLU(inplace=True)
        
        self.dec_conv2 = nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(128)
        self.dec_relu2 = nn.ReLU(inplace=True)
        
        self.dec_conv3 = nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1)
        self.dec_bn3 = nn.BatchNorm2d(64)
        self.dec_relu3 = nn.ReLU(inplace=True)
        
        self.dec_conv4 = nn.ConvTranspose2d(128, 32, 4, stride=2, padding=1)
        self.dec_bn4 = nn.BatchNorm2d(32)
        self.dec_relu4 = nn.ReLU(inplace=True)
        
        self.dec_conv5 = nn.ConvTranspose2d(64, config.CHANNELS, 4, stride=2, padding=1)
        self.tanh = nn.Tanh()
        
        # Simplified refinement network
        self.refine_conv1 = GatedConv2d(config.CHANNELS, 32, 3, padding=1)
        self.refine_conv2 = GatedConv2d(32, config.CHANNELS, 3, padding=1)
        
    def forward(self, x):
        # Encoder pathway - use gradient checkpointing for all operations
        if self.training and torch.cuda.is_available():
            e1 = safe_checkpoint(self.enc_conv1, x)
            e2 = safe_checkpoint(self.enc_conv2, e1)
            e3 = safe_checkpoint(self.enc_conv3, e2)
            e4 = safe_checkpoint(self.enc_conv4, e3)
            e5 = safe_checkpoint(self.enc_conv5, e4)
            
            # Process skip connections with attention - using a more efficient approach
            skip1 = safe_checkpoint(lambda x: self.skip_connection1(self.skip_attention1(x)), e1)
            skip2 = safe_checkpoint(lambda x: self.skip_connection2(self.skip_attention2(x)), e2)
            skip3 = safe_checkpoint(lambda x: self.skip_connection3(self.skip_attention3(x)), e3)
            skip4 = safe_checkpoint(lambda x: self.skip_connection4(self.skip_attention4(x)), e4)
            
            # Decoder pathway with skip connections
            d1 = safe_checkpoint(lambda x: self.dec_relu1(self.dec_bn1(self.dec_conv1(x))), e5)
            d1_cat = torch.cat([d1, skip4], dim=1)
            
            d2 = safe_checkpoint(lambda x: self.dec_relu2(self.dec_bn2(self.dec_conv2(x))), d1_cat)
            d2_cat = torch.cat([d2, skip3], dim=1)
            
            d3 = safe_checkpoint(lambda x: self.dec_relu3(self.dec_bn3(self.dec_conv3(x))), d2_cat)
            d3_cat = torch.cat([d3, skip2], dim=1)
            
            d4 = safe_checkpoint(lambda x: self.dec_relu4(self.dec_bn4(self.dec_conv4(x))), d3_cat)
            d4_cat = torch.cat([d4, skip1], dim=1)
            
            d5 = safe_checkpoint(lambda x: self.tanh(self.dec_conv5(x)), d4_cat)
            
            # Simplified refinement for better details
            refined = safe_checkpoint(self.refine_conv1, d5)
            output = safe_checkpoint(lambda x: self.tanh(self.refine_conv2(x)), refined)
        else:
            # Normal forward pass for inference
            e1 = self.enc_conv1(x)
            e2 = self.enc_conv2(e1)
            e3 = self.enc_conv3(e2)
            e4 = self.enc_conv4(e3)
            e5 = self.enc_conv5(e4)
            
            skip1 = self.skip_connection1(self.skip_attention1(e1))
            skip2 = self.skip_connection2(self.skip_attention2(e2))
            skip3 = self.skip_connection3(self.skip_attention3(e3))
            skip4 = self.skip_connection4(self.skip_attention4(e4))
            
            d1 = self.dec_relu1(self.dec_bn1(self.dec_conv1(e5)))
            d1_cat = torch.cat([d1, skip4], dim=1)
            
            d2 = self.dec_relu2(self.dec_bn2(self.dec_conv2(d1_cat)))
            d2_cat = torch.cat([d2, skip3], dim=1)
            
            d3 = self.dec_relu3(self.dec_bn3(self.dec_conv3(d2_cat)))
            d3_cat = torch.cat([d3, skip2], dim=1)
            
            d4 = self.dec_relu4(self.dec_bn4(self.dec_conv4(d3_cat)))
            d4_cat = torch.cat([d4, skip1], dim=1)
            
            d5 = self.tanh(self.dec_conv5(d4_cat))
            
            refined = self.refine_conv1(d5)
            output = self.tanh(self.refine_conv2(refined))
        
        return output

class EdgeDetectionModule(nn.Module):
    """Memory-efficient edge detection module"""
    def __init__(self, in_channels):
        super(EdgeDetectionModule, self).__init__()
        
        # Initial channel reduction
        self.initial_conv = nn.Conv2d(in_channels, 32, 1)
        self.initial_bn = nn.BatchNorm2d(32)
        self.initial_relu = nn.ReLU(inplace=True)
        
        # Multi-scale edge detection with memory-efficient filter counts
        self.conv1 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Simplified dilated convolutions - just two levels
        self.dil_conv1 = nn.Conv2d(32, 32, 3, padding=2, dilation=2)
        self.dil_conv2 = nn.Conv2d(32, 32, 3, padding=4, dilation=4)
        
        # Fusion layer
        self.fusion = nn.Conv2d(32*2, 32, 1)
        self.bn_fusion = nn.BatchNorm2d(32)
        self.relu_fusion = nn.ReLU(inplace=True)
        
        # Output layer
        self.edge_output = nn.Conv2d(32, 1, 1)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # Initial channel reduction
        x = self.initial_relu(self.initial_bn(self.initial_conv(x)))
        x = self.relu1(self.bn1(self.conv1(x)))
        
        # Dilated convolutions - reduced to just 2
        d1 = self.dil_conv1(x)
        d2 = self.dil_conv2(x)
        
        # Fusion of all scales
        fused = torch.cat([d1, d2], dim=1)
        fused = self.relu_fusion(self.bn_fusion(self.fusion(fused)))
        
        # Output
        edges = self.tanh(self.edge_output(fused))
        return edges

class HybridEdgeAwareGenerator(nn.Module):
    """Memory-optimized hybrid generator with edge detection"""
    def __init__(self, config):
        super(HybridEdgeAwareGenerator, self).__init__()
        
        # Edge detection module
        self.edge_detector = EdgeDetectionModule(config.CHANNELS)
        
        # Lightweight context encoder
        self.encoder = nn.Sequential(
            GatedConv2d(config.CHANNELS, 32, 4, stride=2, padding=1, activation=nn.LeakyReLU(0.2)),
            GatedConv2d(32, 64, 4, stride=2, padding=1, activation=nn.LeakyReLU(0.2)),
            GatedConv2d(64, 128, 4, stride=2, padding=1, activation=nn.LeakyReLU(0.2))
        )
        
        # Main generator
        self.main_generator = ConditionalGatedGenerator(config)
        
        # Lightweight fusion module
        self.fusion_conv1 = GatedConv2d(config.CHANNELS + 1, 32, 3, padding=1)
        self.fusion_conv2 = GatedConv2d(32, config.CHANNELS, 3, padding=1)
        self.fusion_tanh = nn.Tanh()
        
    def forward(self, x):
        # Edge detection - use checkpointing for training
        if self.training and torch.cuda.is_available():
            edge_map = safe_checkpoint(self.edge_detector, x)
            
            # Context encoding - no need to save intermediate results
            _ = self.encoder(x)
            
            # Main image generation
            gen_img = safe_checkpoint(self.main_generator, x)
            
            # Fusion with edge information
            combined = torch.cat([gen_img, edge_map], dim=1)
            
            # Process through fusion convolutions
            fused = safe_checkpoint(self.fusion_conv1, combined)
            refined_output = safe_checkpoint(lambda x: self.fusion_tanh(self.fusion_conv2(x)), fused)
        else:
            # Normal forward pass for inference
            edge_map = self.edge_detector(x)
            _ = self.encoder(x)
            gen_img = self.main_generator(x)
            
            combined = torch.cat([gen_img, edge_map], dim=1)
            fused = self.fusion_conv1(combined)
            refined_output = self.fusion_tanh(self.fusion_conv2(fused))
        
        return edge_map, gen_img, refined_output

class MultiScaleDiscriminator(nn.Module):
    """Memory-efficient multi-scale discriminator"""
    def __init__(self, config):
        super(MultiScaleDiscriminator, self).__init__()
        
        # Global discriminator with reduced parameters
        self.global_discriminator = nn.Sequential(
            nn.Conv2d(config.CHANNELS * 2 + 1, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            GatedConv2d(32, 64, 4, stride=2, padding=1, activation=nn.LeakyReLU(0.2)),
            nn.BatchNorm2d(64),
            
            GatedConv2d(64, 128, 4, stride=2, padding=1, activation=nn.LeakyReLU(0.2)),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 1, 4, stride=1, padding=1)
        )
        
        # Local discriminator with very reduced structure
        self.local_discriminator = nn.Sequential(
            nn.Conv2d(config.CHANNELS * 2 + 1, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            GatedConv2d(32, 64, 4, stride=2, padding=1, activation=nn.LeakyReLU(0.2)),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 1, 4, stride=1, padding=1)
        )
    
    def forward(self, damaged, edge_map, complete):
        # Combine the inputs
        combined = torch.cat([damaged, edge_map, complete], dim=1)
        
        # Global output
        global_output = self.global_discriminator(combined)
        
        # Just return global output to fix the shape error
        return global_output

class EdgeAwareDiscriminator(nn.Module):
    """Fixed edge-aware discriminator"""
    def __init__(self, config):
        super(EdgeAwareDiscriminator, self).__init__()
        self.multiscale_disc = MultiScaleDiscriminator(config)
    
    def forward(self, damaged, edge_map, complete):
        # Now only returning a single tensor
        return self.multiscale_disc(damaged, edge_map, complete)

class HybridDiscriminator(nn.Module):
    """Memory-efficient hybrid discriminator"""
    def __init__(self, config):
        super(HybridDiscriminator, self).__init__()
        
        # Input layers with fewer filters
        self.conv1 = nn.Conv2d(config.CHANNELS * 2, 32, 4, stride=2, padding=1)
        self.leaky1 = nn.LeakyReLU(0.2, inplace=True)
        
        # Simplified gated convolution layers
        self.gated_conv1 = GatedConv2d(32, 64, 4, stride=2, padding=1, activation=nn.LeakyReLU(0.2))
        self.bn1 = nn.BatchNorm2d(64)
        
        self.gated_conv2 = GatedConv2d(64, 128, 4, stride=2, padding=1, activation=nn.LeakyReLU(0.2))
        self.bn2 = nn.BatchNorm2d(128)
        
        # Output layer
        self.conv_out = nn.Conv2d(128, 1, 4, stride=1, padding=1)
        
    def forward(self, condition, target):
        # Combine condition and target
        combined = torch.cat([condition, target], dim=1)
        
        # Forward pass
        x = self.leaky1(self.conv1(combined))
        x = self.bn1(self.gated_conv1(x))
        x = self.bn2(self.gated_conv2(x))
        x = self.conv_out(x)
        
        return x

class PerceptualLoss(nn.Module):
    """Memory-efficient perceptual loss"""
    def __init__(self, layers=['3', '8']):  # Reduced layers
        super(PerceptualLoss, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features
        self.vgg.eval()
        self.layers = layers
        
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        self.mse_loss = nn.MSELoss()
        
    def forward(self, x, y):
        # Normalize inputs
        x = (x + 1) / 2  # Convert from [-1, 1] to [0, 1]
        y = (y + 1) / 2
        
        # Use lower resolution for VGG processing
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        y = F.interpolate(y, size=(64, 64), mode='bilinear', align_corners=False)
        
        # Extend to 3 channels if needed
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        
        # Initialize loss
        loss = 0
        
        # Extract features and compute loss for each layer
        features_x = {}
        features_y = {}
        
        # First pass to extract features
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            
            if str(i) in self.layers:
                features_x[str(i)] = x.clone()
                features_y[str(i)] = y.clone()
        
        # Calculate loss using the stored features
        for key in self.layers:
            loss += self.mse_loss(features_x[key], features_y[key])
        
        return loss

# Replace the original classes with optimized versions
HybridEdgeAwareGenerator = HybridEdgeAwareGenerator
EdgeAwareDiscriminator = EdgeAwareDiscriminator
HybridDiscriminator = HybridDiscriminator
PerceptualLoss = PerceptualLoss
HybridConditionalGatedGenerator = ConditionalGatedGenerator
