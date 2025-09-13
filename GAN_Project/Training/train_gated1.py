import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.data_utils import ArtifactDataset
from models.gated_gan import GatedGenerator, GatedDiscriminator, GatedEdgeGenerator, GatedEdgeDiscriminator
from utils.training_monitor import TrainingMonitor
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from datetime import datetime

def denormalize(tensor):
    return (tensor + 1) / 2

def save_sample_images(generator, dataloader, device, save_dir, epoch, num_samples=5):
    """Save sample images during training"""
    generator.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        real_complete = batch['complete'].to(device)
        real_damaged = batch['damaged'].to(device)
        
        if isinstance(generator, GatedEdgeGenerator):
            edge_map, gen_complete = generator(real_damaged)
            edge_map_denorm = denormalize(edge_map)
        else:
            gen_complete = generator(real_damaged)
            
        real_complete_denorm = denormalize(real_complete)
        real_damaged_denorm = denormalize(real_damaged)
        gen_complete_denorm = denormalize(gen_complete)
        
        for i in range(min(num_samples, real_complete.size(0))):
            # Create comparison grid
            if isinstance(generator, GatedEdgeGenerator):
                comparison = torch.cat([
                    real_damaged_denorm[i],  # Damaged
                    edge_map_denorm[i].repeat(3, 1, 1),  # Edge map (convert to 3 channels)
                    gen_complete_denorm[i],  # Generated
                    real_complete_denorm[i]  # Ground truth
                ], dim=2)
            else:
                comparison = torch.cat([
                    real_damaged_denorm[i],  # Damaged
                    gen_complete_denorm[i],  # Generated
                    real_complete_denorm[i]  # Ground truth
                ], dim=2)
                
            filename = f'epoch_{epoch}_sample_{i}.png'
            save_image(comparison, os.path.join(save_dir, filename))
    
    generator.train()

def save_reconstructed_images(generator, dataloader, device, save_dir, epoch, num_samples=10):
    """ Save final reconstructed images after last epoch """
    generator.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        real_complete = batch['complete'].to(device)
        real_damaged = batch['damaged'].to(device)
        
        if isinstance(generator, GatedEdgeGenerator):
            _, gen_complete = generator(real_damaged)
        else:
            gen_complete = generator(real_damaged)

        gen_complete_denorm = denormalize(gen_complete)
        
        for i in range(min(num_samples, real_complete.size(0))):
            filename = f'final_reconstructed_epoch{epoch}_sample{i}.png'
            save_image(gen_complete_denorm[i], os.path.join(save_dir, filename))
        
        print(f"✅ Saved {min(num_samples, real_complete.size(0))} final reconstructed images.")

def train_gated_conv(config, use_edge_model=True):
    print("🔄 Initializing Gated Convolution GAN training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # ✅ EXACT same folder structure as EdgeConnect
    model_name = 'gated_edge_gan' if use_edge_model else 'gated_conv_gan'
    fixed_results_dir = os.path.join('src', 'results1', 'gated_gan')

    os.makedirs(fixed_results_dir, exist_ok=True)

    reconstructed_dir = os.path.join(fixed_results_dir, 'reconstructed_images')
    checkpoint_dir = os.path.join(fixed_results_dir, 'checkpoints')
    samples_dir = os.path.join(fixed_results_dir, 'samples')

    os.makedirs(reconstructed_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    print(f"📁 Results will be saved to: {fixed_results_dir}")

    # Load dataset
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

    # Initialize models based on the selected architecture
    if use_edge_model:
        generator = GatedEdgeGenerator(config).to(device)
        discriminator = GatedEdgeDiscriminator(config).to(device)
        print("🔄 Using GatedEdgeGenerator with edge detection")
    else:
        generator = GatedGenerator(config).to(device)
        discriminator = GatedDiscriminator(config).to(device)
        print("🔄 Using standard GatedGenerator")

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))

    # Loss functions
    criterion_GAN = nn.BCELoss()
    criterion_pixel = nn.L1Loss()

    # Track losses
    g_losses, d_losses = [], []

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

            # -----------------------
            # Train Discriminator
            # -----------------------
            optimizer_D.zero_grad()
            
            # Forward pass depends on generator type
            if use_edge_model:
                edge_map, fake_complete = generator(real_damaged)
                real_validity = discriminator(real_complete, edge_map, real_damaged)
                fake_validity = discriminator(fake_complete.detach(), edge_map.detach(), real_damaged)
            else:
                fake_complete = generator(real_damaged)
                real_validity = discriminator(real_complete, real_damaged)
                fake_validity = discriminator(fake_complete.detach(), real_damaged)

            # Compute discriminator loss
            d_loss_real = criterion_GAN(real_validity, torch.ones_like(real_validity))
            d_loss_fake = criterion_GAN(fake_validity, torch.zeros_like(fake_validity))
            d_loss = (d_loss_real + d_loss_fake) * 0.5
            d_loss.backward()
            optimizer_D.step()

            # -----------------------
            # Train Generator
            # -----------------------
            optimizer_G.zero_grad()
            
            # Forward pass depends on generator type
            if use_edge_model:
                # Recompute outputs for backprop
                edge_map, fake_complete = generator(real_damaged)
                gen_validity = discriminator(fake_complete, edge_map, real_damaged)
                
                # Edge loss (comparing generated edge map to some target if available)
                # Here we use a simple approximation: apply a Sobel filter to the real complete image
                # A more sophisticated approach would be to use a pre-trained edge detector
                g_loss_pixel = criterion_pixel(fake_complete, real_complete)
                g_loss_edge = criterion_pixel(edge_map, torch.zeros_like(edge_map))  # Simplified edge loss
                
                g_loss_gan = criterion_GAN(gen_validity, torch.ones_like(gen_validity))
                g_loss = (g_loss_gan * config.LAMBDA_ADV + 
                          g_loss_pixel * config.LAMBDA_PIXEL + 
                          g_loss_edge * config.LAMBDA_STYLE)  # Using LAMBDA_STYLE weight for edge loss
                
            else:
                # Standard GatedGenerator
                fake_complete = generator(real_damaged)
                gen_validity = discriminator(fake_complete, real_damaged)
                g_loss_gan = criterion_GAN(gen_validity, torch.ones_like(gen_validity))
                g_loss_pixel = criterion_pixel(fake_complete, real_complete)
                g_loss = g_loss_gan * config.LAMBDA_ADV + g_loss_pixel * config.LAMBDA_PIXEL
            
            g_loss.backward()
            optimizer_G.step()

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()

            # Print progress
            if batch_idx % max(1, total_batches // 10) == 0:
                print(f"[Epoch {epoch+1}/{config.NUM_EPOCHS}] "
                      f"[Batch {batch_idx}/{total_batches}] "
                      f"[D loss: {d_loss.item():.4f}] "
                      f"[G loss: {g_loss.item():.4f}]")

        # Track average losses
        avg_g_loss = total_g_loss / total_batches
        avg_d_loss = total_d_loss / total_batches
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        # Save sample images at each epoch
        save_sample_images(generator, dataloader, device, samples_dir, epoch+1, 
                          num_samples=config.NUM_DISPLAY_SAMPLES)

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

    print("✅ Training completed!")

    # Save final loss plot
    loss_plot_path = os.path.join(fixed_results_dir, 'loss_plot.png')
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"📈 Loss plot saved to {loss_plot_path}")

    # Save final model
    final_model_path = os.path.join(fixed_results_dir, 'final_model.pth')
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
    }, final_model_path)
    print(f"💾 Saved final model to {final_model_path}")

    # Save final reconstructed images
    save_reconstructed_images(generator, dataloader, device, reconstructed_dir, config.NUM_EPOCHS)

if __name__ == "__main__":
    from configs.config import Config
    config = Config()
    
    # Set to True to use the edge-aware model, False to use standard GatedGenerator
    use_edge_model = True
    train_gated_conv(config, use_edge_model)
