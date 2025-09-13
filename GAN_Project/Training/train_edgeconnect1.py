import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.data_utils import ArtifactDataset
from models.edge_connect_gan import EdgeGenerator, EdgeDiscriminator
from utils.training_monitor import TrainingMonitor
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from datetime import datetime

def denormalize(tensor):
    return (tensor + 1) / 2

def save_loss_plot(g_losses, d_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss plot saved to {save_path}")

def save_reconstructed_images(generator, dataloader, device, save_dir, epoch, num_samples=10):
    """
    Save reconstructed images from the generator to disk
    """
    generator.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        real_complete = batch['complete'].to(device)
        real_damaged = batch['damaged'].to(device)
        edge_map, gen_complete = generator(real_damaged)

        # Denormalize images for saving
        gen_complete_denorm = denormalize(gen_complete)
        
        # Save reconstructed images
        for i in range(min(num_samples, real_complete.size(0))):
            # Create filename with epoch and sample number
            filename = f'reconstructed_epoch{epoch}_sample{i}.png'
            
            # Save reconstructed image
            save_image(gen_complete_denorm[i], os.path.join(save_dir, filename))
        
        print(f"✅ Saved {min(num_samples, real_complete.size(0))} reconstructed images for epoch {epoch}")

def display_sample_images(generator, dataloader, device, save_dir, epoch, num_samples=3):
    """
    Display and save sample comparison images
    """
    generator.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        real_complete = batch['complete'].to(device)
        real_damaged = batch['damaged'].to(device)
        edge_map, gen_complete = generator(real_damaged)

        # Denormalize images
        real_complete = denormalize(real_complete)
        real_damaged = denormalize(real_damaged)
        gen_complete = denormalize(gen_complete)
        edge_map = denormalize(edge_map)

        # Create a figure to save
        fig, axs = plt.subplots(min(num_samples, real_complete.size(0)), 4, figsize=(16, 4 * min(num_samples, real_complete.size(0))))
        fig.suptitle(f"EdgeConnect GAN - Epoch {epoch}", fontsize=16)
        
        # Handle case when only one sample
        if num_samples == 1:
            axs = [axs]

        for j in range(min(num_samples, real_complete.size(0))):
            axs[j][0].imshow(real_damaged[j].permute(1, 2, 0).cpu().numpy())
            axs[j][0].set_title("Damaged")
            axs[j][0].axis('off')

            axs[j][1].imshow(edge_map[j].squeeze().cpu().numpy(), cmap='gray')
            axs[j][1].set_title("Edges")
            axs[j][1].axis('off')

            axs[j][2].imshow(gen_complete[j].permute(1, 2, 0).cpu().numpy())
            axs[j][2].set_title("Reconstructed")
            axs[j][2].axis('off')

            axs[j][3].imshow(real_complete[j].permute(1, 2, 0).cpu().numpy())
            axs[j][3].set_title("Original")
            axs[j][3].axis('off')

        plt.tight_layout()
        
        # Save the figure
        comparison_path = os.path.join(save_dir, f'comparison_epoch{epoch}.png')
        plt.savefig(comparison_path)
        plt.close()
        
        print(f"✅ Saved comparison image for epoch {epoch}")

def detect_edges(image_tensor):
    image = image_tensor.cpu().numpy().transpose(1, 2, 0)
    image = ((image + 1) * 127.5).astype(np.uint8)
    edges = cv2.Canny(image, 100, 200)
    edges = torch.from_numpy(edges).float() / 255.0
    edges = edges.unsqueeze(0)
    return 2.0 * edges - 1.0

def train_edge_connect(config):
    print("🔄 Initializing EdgeConnect GAN training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    
    # Override the results directory to use a fixed path with no timestamp
    # This is the key change - use the direct path to edge_connect_gan
    fixed_results_dir = os.path.join('src', 'results1', 'edgeconnect_gan')
    
    # Create main directory for this GAN directly
    os.makedirs(fixed_results_dir, exist_ok=True)
    
    # Create subdirectories
    reconstructed_dir = os.path.join(fixed_results_dir, 'reconstructed_images')
    checkpoint_dir = os.path.join(fixed_results_dir, 'checkpoints')
    samples_dir = os.path.join(fixed_results_dir, 'samples')
    
    os.makedirs(reconstructed_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    
    print(f"📁 Results will be saved to: {fixed_results_dir}")

    # Set up dataset and dataloader
    dataset = ArtifactDataset(
        complete_dir=config.COMPLETE_DIR,
        damaged_dir=config.DAMAGED_DIR,
        image_size=config.IMAGE_SIZE
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    print(f"📊 Dataset loaded with {len(dataset)} images")

    # Initialize models
    generator = EdgeGenerator(config).to(device)
    discriminator = EdgeDiscriminator(config).to(device)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))

    # Loss functions
    criterion_GAN = nn.BCELoss()
    criterion_pixel = nn.L1Loss()
    criterion_edge = nn.MSELoss()

    # Lists to track losses
    g_losses = []
    d_losses = []

    print("\n🚀 Starting training loop...")
    total_batches = len(dataloader)

    for epoch in range(config.NUM_EPOCHS):
        generator.train()
        discriminator.train()

        total_g_loss = 0
        total_d_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            real_complete = batch['complete'].to(device)
            real_damaged = batch['damaged'].to(device)
            batch_size = real_complete.size(0)

            # Detect edges from real images
            real_edges = torch.stack([detect_edges(img) for img in real_complete]).to(device)

            # Generate fake edges and completed images
            fake_edges, fake_complete = generator(real_damaged)
            
            # -----------------------
            # Train Discriminator
            # -----------------------
            optimizer_D.zero_grad()
            
            # Process real images
            real_validity = discriminator(real_damaged, real_edges, real_complete)
            
            # Process fake images
            fake_validity = discriminator(real_damaged, fake_edges.detach(), fake_complete.detach())
            
            # Calculate loss
            d_loss_real = criterion_GAN(real_validity, torch.ones_like(real_validity))
            d_loss_fake = criterion_GAN(fake_validity, torch.zeros_like(fake_validity))
            d_loss = (d_loss_real + d_loss_fake) * 0.5
            
            # Backpropagate and update
            d_loss.backward()
            optimizer_D.step()

            # -----------------------
            # Train Generator
            # -----------------------
            optimizer_G.zero_grad()
            
            # Calculate adversarial loss
            gen_validity = discriminator(real_damaged, fake_edges, fake_complete)
            g_loss_gan = criterion_GAN(gen_validity, torch.ones_like(gen_validity))
            
            # Calculate pixel loss
            g_loss_pixel = criterion_pixel(fake_complete, real_complete)
            
            # Calculate edge loss
            g_loss_edge = criterion_edge(fake_edges, real_edges)
            
            # Combine losses
            g_loss = (g_loss_gan * config.LAMBDA_ADV + 
                     g_loss_pixel * config.LAMBDA_PIXEL + 
                     g_loss_edge * config.LAMBDA_STYLE)
            
            # Backpropagate and update
            g_loss.backward()
            optimizer_G.step()

            # Track losses
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

            # Print progress
            if batch_idx % max(1, total_batches // 10) == 0:
                print(f"[Epoch {epoch+1}/{config.NUM_EPOCHS}] "
                      f"[Batch {batch_idx}/{total_batches}] "
                      f"[D loss: {d_loss.item():.4f}] "
                      f"[G loss: {g_loss.item():.4f}]")

        # Calculate average losses for this epoch
        avg_g_loss = total_g_loss / total_batches
        avg_d_loss = total_d_loss / total_batches
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        print(f"\n✅ Epoch {epoch+1}/{config.NUM_EPOCHS} completed.")
        print(f"📉 Avg Losses - Generator: {avg_g_loss:.4f}, Discriminator: {avg_d_loss:.4f}")

        # Save checkpoint at specified intervals
        if (epoch + 1) % config.SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'g_loss': avg_g_loss,
                'd_loss': avg_d_loss
            }, checkpoint_path)
            print(f"💾 Saved checkpoint at epoch {epoch+1}")
            
        # Only save images after the final epoch
        if epoch + 1 == config.NUM_EPOCHS:
            # Save reconstructed images only after the final epoch
            save_reconstructed_images(
                generator, 
                dataloader, 
                device, 
                reconstructed_dir, 
                epoch + 1, 
                num_samples=config.NUM_DISPLAY_SAMPLES
            )
            
            # Save sample comparisons only after the final epoch
            display_sample_images(
                generator, 
                dataloader, 
                device, 
                samples_dir,
                epoch + 1, 
                num_samples=min(3, config.NUM_DISPLAY_SAMPLES)
            )

    print("✅ Training completed!")

    # Save final loss plot
    loss_plot_path = os.path.join(fixed_results_dir, 'loss_plot.png')
    save_loss_plot(g_losses, d_losses, loss_plot_path)
    
    # Save final model
    final_model_path = os.path.join(fixed_results_dir, 'final_model.pth')
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
    }, final_model_path)
    print(f"💾 Saved final model to {final_model_path}")
    
    # Generate final batch of samples
    print("🖼️ Generating final samples...")
    display_sample_images(
        generator, 
        dataloader, 
        device, 
        samples_dir,
        'final', 
        num_samples=6
    )
    
    return fixed_results_dir  # Return the output directory for reference

if __name__ == "__main__":
    from configs.config import Config
    config = Config()
    train_edge_connect(config)
