import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

class TrainingMonitor:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.samples_dir = os.path.join(save_dir, 'samples')
        self.plots_dir = os.path.join(save_dir, 'plots')
        
        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
    def denormalize(self, tensor):
        """Denormalize the tensor from [-1, 1] to [0, 1] range"""
        return (tensor + 1) / 2
    
    def tensor_to_numpy(self, tensor):
        """Convert a PyTorch tensor to a NumPy array."""
        tensor = self.denormalize(tensor)
        tensor = torch.clamp(tensor, 0, 1)
        
        if len(tensor.shape) == 4:
            tensor = make_grid(tensor, nrow=1, normalize=False)
        return tensor.detach().cpu().numpy().transpose(1, 2, 0)
    
    def save_sample_images(self, real_complete, real_damaged, fake_complete, epoch, sample_idx=0):
        """Save sample images during training."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Epoch {epoch+1} - Sample {sample_idx+1}')
        
        images = [real_damaged, fake_complete, real_complete]
        titles = ['Damaged Input', 'Generated Restoration', 'Ground Truth']
        
        for ax, img, title in zip(axes, images, titles):
            if len(img.shape) == 4:
                img = img[0]  
            np_img = self.tensor_to_numpy(img)
            ax.imshow(np_img)
            ax.set_title(title)
            ax.axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(self.samples_dir, f'epoch_{epoch+1:04d}_sample_{sample_idx+1}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    
    def save_loss_plot(self, g_losses, d_losses, epoch):
        """Save loss plot."""
        plt.figure(figsize=(10, 5))
        plt.plot(g_losses, label='Generator Loss')
        plt.plot(d_losses, label='Discriminator Loss')
        plt.title(f'Training Losses (up to Epoch {epoch+1})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(self.plots_dir, f'losses_epoch_{epoch+1:04d}.png')
        plt.savefig(save_path)
        plt.close()
        
        # Also save a CSV of the losses for further analysis
        import csv
        csv_path = os.path.join(self.plots_dir, 'training_losses.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Generator Loss', 'Discriminator Loss'])
            for i in range(len(g_losses)):
                writer.writerow([i+1, g_losses[i], d_losses[i]])
