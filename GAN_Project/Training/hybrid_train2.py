import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.data_utils import ArtifactDataset
from models.hybrid_gan2 import HybridConditionalGatedGenerator, HybridDiscriminator, HybridEdgeAwareGenerator, EdgeAwareDiscriminator
from models.hybrid_gan2 import PerceptualLoss
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def denormalize(tensor):
    """Convert normalized [-1, 1] tensor to [0, 1] for visualization"""
    return (tensor + 1) / 2

def save_loss_plot(g_losses, d_losses, save_path):
    """Save loss curves to disk"""
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
    print(f"📈 Loss plot saved to {save_path}")

def save_sample_images(generator, dataloader, device, save_dir, epoch, num_samples=5, edge_aware=False):
    """Save sample images during training"""
    generator.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        real_complete = batch['complete'].to(device)
        real_damaged = batch['damaged'].to(device)
        
        if edge_aware:
            edge_map, gen_img, refined_output = generator(real_damaged)
            edge_map_denorm = denormalize(edge_map)
            gen_img_denorm = denormalize(gen_img)
            refined_denorm = denormalize(refined_output)
        else:
            gen_complete = generator(real_damaged)
            gen_complete_denorm = denormalize(gen_complete)
            
        real_complete_denorm = denormalize(real_complete)
        real_damaged_denorm = denormalize(real_damaged)
        
        for i in range(min(num_samples, real_complete.size(0))):
            # Create comparison grid
            if edge_aware:
                comparison = torch.cat([
                    real_damaged_denorm[i],  # Damaged
                    edge_map_denorm[i].repeat(3, 1, 1) if edge_map_denorm[i].size(0) == 1 else edge_map_denorm[i],  # Edge map (convert to 3 channels if needed)
                    gen_img_denorm[i],  # Initial generated
                    refined_denorm[i],  # Edge-refined 
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

def save_reconstructed_images(generator, dataloader, device, save_dir, epoch, num_samples=10, edge_aware=False):
    """Save final reconstructed images after training"""
    generator.eval()
    with torch.no_grad():
        batch = next(iter(dataloader))
        real_complete = batch['complete'].to(device)
        real_damaged = batch['damaged'].to(device)
        
        if edge_aware:
            _, _, refined_output = generator(real_damaged)
            final_output = refined_output
        else:
            final_output = generator(real_damaged)
        
        final_output_denorm = denormalize(final_output)
        
        for i in range(min(num_samples, real_complete.size(0))):
            filename = f'final_reconstructed_epoch{epoch}_sample{i}.png'
            save_image(final_output_denorm[i], os.path.join(save_dir, filename))
        
        print(f"✅ Saved {min(num_samples, real_complete.size(0))} final reconstructed images.")

def train_hybrid_gan(config, use_edge_aware=True):
    """Main training function for the hybrid GAN models"""
    print("🔄 Initializing Hybrid Conditional-Gated GAN training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # Create results directory structure
    model_name = 'hybrid_edge_conditional_gated_gan2' if use_edge_aware else 'hybrid_conditional_gated_gan'
    fixed_results_dir = os.path.join('src', 'results1', 'hybrid_conditional_gated_gan')


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

    # Initialize models based on selected architecture
    if use_edge_aware:
        generator = HybridEdgeAwareGenerator(config).to(device)
        discriminator = EdgeAwareDiscriminator(config).to(device)
        print("🔄 Using HybridEdgeAwareGenerator with edge detection")
    else:
        generator = HybridConditionalGatedGenerator(config).to(device)
        discriminator = HybridDiscriminator(config).to(device)
        print("🔄 Using HybridConditionalGatedGenerator")

    # Initialize losses
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_pixel = nn.L1Loss()
    criterion_edge = nn.L1Loss()
    perceptual_criterion = PerceptualLoss().to(device)

    # Initialize optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))

    # Track losses
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

            # -----------------------
            # Train Discriminator
            # -----------------------
            optimizer_D.zero_grad()
            
            # Forward pass depends on generator type
            if use_edge_aware:
                edge_map, gen_img, refined_output = generator(real_damaged)
                
                # Create real and fake labels
                valid_shape = discriminator(real_damaged, edge_map, real_complete).shape
                valid = torch.ones(valid_shape, device=device)
                fake = torch.zeros(valid_shape, device=device)
                
                # Train on real images
                real_validity = discriminator(real_damaged, edge_map, real_complete)
                d_real_loss = criterion_GAN(real_validity, valid)
                
                # Train on fake images
                fake_validity = discriminator(real_damaged, edge_map.detach(), refined_output.detach())
                d_fake_loss = criterion_GAN(fake_validity, fake)
                
            else:
                # Standard hybrid generator
                fake_complete = generator(real_damaged)
                
                # Create real and fake labels
                valid_shape = discriminator(real_damaged, real_complete).shape
                valid = torch.ones(valid_shape, device=device)
                fake = torch.zeros(valid_shape, device=device)
                
                # Train on real images
                real_validity = discriminator(real_damaged, real_complete)
                d_real_loss = criterion_GAN(real_validity, valid)
                
                # Train on fake images
                fake_validity = discriminator(real_damaged, fake_complete.detach())
                d_fake_loss = criterion_GAN(fake_validity, fake)
            
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) * 0.5
            d_loss.backward()
            optimizer_D.step()

            # -----------------------
            # Train Generator
            # -----------------------
            optimizer_G.zero_grad()
            
            # Forward pass depends on generator type
            if use_edge_aware:
                # Process through generator again for backprop
                edge_map, gen_img, refined_output = generator(real_damaged)
                
                # Adversarial loss
                gen_validity = discriminator(real_damaged, edge_map, refined_output)
                g_loss_gan = criterion_GAN(gen_validity, valid)
                
                # Content losses for different outputs
                g_loss_edge = criterion_edge(edge_map, torch.zeros_like(edge_map)) * 0.1  # Simple edge loss
                g_loss_content = criterion_pixel(gen_img, real_complete)
                g_loss_refined = criterion_pixel(refined_output, real_complete)
                
                # Perceptual loss
                g_loss_perceptual = perceptual_criterion(refined_output, real_complete)
                
                # Compute total generator loss
                g_loss = (g_loss_gan * config.LAMBDA_ADV + 
                         g_loss_content * config.LAMBDA_PIXEL + 
                         g_loss_refined * config.LAMBDA_PIXEL * 1.5 +  # Higher weight for refined output
                         g_loss_edge * config.LAMBDA_STYLE +  # Edge loss with style weight
                         g_loss_perceptual * config.LAMBDA_PERCEPTUAL)  # Added perceptual loss
                
            else:
                # Standard hybrid generator
                fake_complete = generator(real_damaged)
                
                # Adversarial loss
                gen_validity = discriminator(real_damaged, fake_complete)
                g_loss_gan = criterion_GAN(gen_validity, valid)
                
                # Pixel-wise loss
                g_loss_pixel = criterion_pixel(fake_complete, real_complete)
                
                # Perceptual loss
                g_loss_perceptual = perceptual_criterion(fake_complete, real_complete)
                
                # Total generator loss
                g_loss = g_loss_gan * config.LAMBDA_ADV + g_loss_pixel * config.LAMBDA_PIXEL + g_loss_perceptual * config.LAMBDA_PERCEPTUAL
            
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

        # Track average losses
        avg_g_loss = total_g_loss / total_batches
        avg_d_loss = total_d_loss / total_batches
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        # Save sample images at each epoch
        save_sample_images(generator, dataloader, device, samples_dir, epoch+1, 
                          num_samples=config.NUM_DISPLAY_SAMPLES,
                          edge_aware=use_edge_aware)

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
    save_loss_plot(g_losses, d_losses, loss_plot_path)

    # Save final model
    final_model_path = os.path.join(fixed_results_dir, 'final_model.pth')
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
    }, final_model_path)
    print(f"💾 Saved final model to {final_model_path}")

    # Save final reconstructed images
    save_reconstructed_images(
        generator, 
        dataloader, 
        device, 
        reconstructed_dir, 
        config.NUM_EPOCHS,
        edge_aware=use_edge_aware
    )
    
    return fixed_results_dir

if __name__ == "__main__":
    from configs.config import Config
    config = Config()
    
    # Set to True to use the edge-aware hybrid model, False to use standard hybrid generator
    use_edge_aware = True
    train_hybrid_gan(config, use_edge_aware)

