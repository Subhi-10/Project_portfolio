import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.data_utils import ArtifactDataset
from models.stylegan3_gan import StyleGAN3Generator, StyleGAN3Discriminator
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import numpy as np

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
        
        # Process in smaller chunks to save memory
        batch_size = real_complete.size(0)
        chunk_size = min(4, batch_size)  # Process max 4 images at once
        
        gen_complete_list = []
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            chunk_damaged = real_damaged[i:end_idx]
            chunk_complete = generator(chunk_damaged)
            gen_complete_list.append(chunk_complete)
            
            # Free memory
            torch.cuda.empty_cache()
            
        gen_complete = torch.cat(gen_complete_list, dim=0)

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
        
        # Process in smaller chunks to save memory
        batch_size = real_complete.size(0)
        chunk_size = min(4, batch_size)  # Process max 4 images at once
        
        gen_complete_list = []
        for i in range(0, min(num_samples, batch_size), chunk_size):
            end_idx = min(i + chunk_size, min(num_samples, batch_size))
            chunk_damaged = real_damaged[i:end_idx]
            chunk_complete = generator(chunk_damaged)
            gen_complete_list.append(chunk_complete)
            
            # Free memory
            torch.cuda.empty_cache()
            
        gen_complete = torch.cat(gen_complete_list, dim=0)

        # Denormalize images
        real_complete = denormalize(real_complete[:num_samples])
        real_damaged = denormalize(real_damaged[:num_samples])
        gen_complete = denormalize(gen_complete)

        # Create a figure to save
        fig, axs = plt.subplots(min(num_samples, real_complete.size(0)), 3, figsize=(15, 4 * min(num_samples, real_complete.size(0))))
        fig.suptitle(f"StyleGAN3 Restoration - Epoch {epoch}", fontsize=16)
        
        # Handle case when only one sample
        if num_samples == 1:
            axs = [axs]

        for j in range(min(num_samples, real_complete.size(0))):
            axs[j][0].imshow(real_damaged[j].permute(1, 2, 0).cpu().numpy())
            axs[j][0].set_title("Damaged")
            axs[j][0].axis('off')

            axs[j][1].imshow(gen_complete[j].permute(1, 2, 0).cpu().numpy())
            axs[j][1].set_title("Reconstructed")
            axs[j][1].axis('off')

            axs[j][2].imshow(real_complete[j].permute(1, 2, 0).cpu().numpy())
            axs[j][2].set_title("Original")
            axs[j][2].axis('off')

        plt.tight_layout()
        
        # Save the figure
        comparison_path = os.path.join(save_dir, f'comparison_epoch{epoch}.png')
        plt.savefig(comparison_path)
        plt.close()
        
        print(f"✅ Saved comparison image for epoch {epoch}")

def load_checkpoint(generator, discriminator, optimizer_G, optimizer_D, checkpoint_path):
    if os.path.exists(checkpoint_path):
        print(f"✅ Resuming training from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        generator.load_state_dict(checkpoint['generator_state_dict'], strict=False)
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'], strict=False)
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        return checkpoint['epoch'] + 1, checkpoint.get('g_losses', []), checkpoint.get('d_losses', [])
    return 0, [], []

def train_stylegan3(config):
    print("🔄 Initializing StyleGAN3 training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    
    # Use a fixed path with no timestamp for StyleGAN3
    fixed_results_dir = os.path.join('src', 'results1', 'stylegan3_gan')
    
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
    print(f"📸 Images will be saved only at the final epoch")

    # Set up dataset and dataloader
    dataset = ArtifactDataset(
        complete_dir=config.COMPLETE_DIR,
        damaged_dir=config.DAMAGED_DIR,
        image_size=config.IMAGE_SIZE
    )
    
    # Keep the original batch size as requested
    dataloader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    print(f"📊 Dataset loaded with {len(dataset)} images")

    # Initialize models and optimizers
    generator = StyleGAN3Generator(config).to(device)
    discriminator = StyleGAN3Discriminator(config).to(device)
    
    # Memory optimization techniques
    # 1. Enable gradient checkpointing to save memory during backprop
    if hasattr(generator, 'use_checkpoint'):
        generator.use_checkpoint = True
    
    # 2. Custom memory cleaning hook
    def clean_memory_hook(module, grad_in, grad_out):
        torch.cuda.empty_cache()
        return None
    
    # Apply memory hook to larger modules
    for module in generator.modules():
        if isinstance(module, nn.Conv2d) and module.out_channels > 128:
            module.register_full_backward_hook(clean_memory_hook)
    
    # Use the original Adam optimizer with parameters from config
    optimizer_G = optim.Adam(generator.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))

    # Check for resuming from checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
    start_epoch, g_losses, d_losses = load_checkpoint(generator, discriminator, optimizer_G, optimizer_D, checkpoint_path)
    
    # Loss functions
    criterion_GAN = nn.BCELoss()
    criterion_pixel = nn.L1Loss()
    
    print("\n🚀 Starting training loop...")
    total_batches = len(dataloader)
    
    for epoch in range(start_epoch, config.NUM_EPOCHS):
        generator.train()
        discriminator.train()
        
        total_g_loss = 0
        total_d_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            real_complete = batch['complete'].to(device)
            real_damaged = batch['damaged'].to(device)
            batch_size = real_complete.size(0)
            
            # Clear cache before processing batch
            torch.cuda.empty_cache()
            
            # -----------------------
            # Train Generator
            # -----------------------
            optimizer_G.zero_grad()
            
            # Process generator in chunks if batch size is large
            if batch_size > 8:
                chunk_size = 8
                g_loss = 0
                gen_complete_list = []
                
                for i in range(0, batch_size, chunk_size):
                    end_idx = min(i + chunk_size, batch_size)
                    
                    # Process chunk
                    chunk_damaged = real_damaged[i:end_idx]
                    chunk_real = real_complete[i:end_idx]
                    chunk_gen = generator(chunk_damaged)
                    gen_complete_list.append(chunk_gen)
                    
                    # Calculate loss for this chunk
                    pred_fake_chunk = discriminator(chunk_damaged, chunk_gen)
                    valid_chunk = torch.ones_like(pred_fake_chunk, device=device)
                    
                    # Combine adversarial and pixel-wise loss
                    chunk_g_loss = criterion_GAN(pred_fake_chunk, valid_chunk) + \
                                  config.LAMBDA_PIXEL * criterion_pixel(chunk_gen, chunk_real)
                    
                    # Scale the loss and backpropagate
                    (chunk_g_loss * (end_idx - i) / batch_size).backward()
                    g_loss += chunk_g_loss.item() * (end_idx - i) / batch_size
                    
                # Combine all generated chunks
                gen_complete = torch.cat(gen_complete_list, dim=0)
            else:
                # For smaller batches, process normally
                gen_complete = generator(real_damaged)
                
                # Calculate adversarial loss
                pred_fake = discriminator(real_damaged, gen_complete)
                valid = torch.ones_like(pred_fake, device=device)
                
                # Combine adversarial and pixel-wise loss
                g_loss = criterion_GAN(pred_fake, valid) + config.LAMBDA_PIXEL * criterion_pixel(gen_complete, real_complete)
                
                # Backpropagate
                g_loss.backward()
            
            # Update generator
            optimizer_G.step()
            
            # Clear memory
            torch.cuda.empty_cache()
            
            # -----------------------
            # Train Discriminator
            # -----------------------
            optimizer_D.zero_grad()
            
            # Process discriminator in chunks if batch size is large
            if batch_size > 8:
                chunk_size = 8
                d_loss = 0
                
                for i in range(0, batch_size, chunk_size):
                    end_idx = min(i + chunk_size, batch_size)
                    
                    # Process chunk
                    chunk_damaged = real_damaged[i:end_idx]
                    chunk_real = real_complete[i:end_idx]
                    chunk_gen = gen_complete[i:end_idx].detach()
                    
                    # Real images
                    pred_real_chunk = discriminator(chunk_damaged, chunk_real)
                    valid_chunk = torch.ones_like(pred_real_chunk, device=device)
                    loss_D_real_chunk = criterion_GAN(pred_real_chunk, valid_chunk)
                    
                    # Fake images
                    pred_fake_chunk = discriminator(chunk_damaged, chunk_gen)
                    fake_chunk = torch.zeros_like(pred_fake_chunk, device=device)
                    loss_D_fake_chunk = criterion_GAN(pred_fake_chunk, fake_chunk)
                    
                    # Combine losses
                    chunk_d_loss = 0.5 * (loss_D_real_chunk + loss_D_fake_chunk)
                    
                    # Scale loss and backpropagate
                    (chunk_d_loss * (end_idx - i) / batch_size).backward()
                    d_loss += chunk_d_loss.item() * (end_idx - i) / batch_size
            else:
                # Process real images
                pred_real = discriminator(real_damaged, real_complete)
                valid = torch.ones_like(pred_real, device=device)
                loss_D_real = criterion_GAN(pred_real, valid)
                
                # Process fake images
                fake = torch.zeros_like(pred_real, device=device)
                loss_D_fake = criterion_GAN(discriminator(real_damaged, gen_complete.detach()), fake)
                
                # Combine losses
                d_loss = 0.5 * (loss_D_real + loss_D_fake)
                
                # Backpropagate
                d_loss.backward()
            
            # Update discriminator
            optimizer_D.step()
            
            # Clear memory
            del gen_complete
            torch.cuda.empty_cache()
            
            # Track losses
            if isinstance(g_loss, torch.Tensor):
                g_loss = g_loss.item()
            if isinstance(d_loss, torch.Tensor):
                d_loss = d_loss.item()
                
            total_g_loss += g_loss
            total_d_loss += d_loss
            
            # Print progress
            if batch_idx % max(1, total_batches // 10) == 0:
                print(f"[Epoch {epoch+1}/{config.NUM_EPOCHS}] "
                      f"[Batch {batch_idx}/{total_batches}] "
                      f"[D loss: {d_loss:.4f}] "
                      f"[G loss: {g_loss:.4f}]")
        
        # Calculate average losses for this epoch
        avg_g_loss = total_g_loss / total_batches
        avg_d_loss = total_d_loss / total_batches
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        print(f"\n✅ Epoch {epoch+1}/{config.NUM_EPOCHS} completed.")
        print(f"📉 Avg Losses - Generator: {avg_g_loss:.4f}, Discriminator: {avg_d_loss:.4f}")
        
        # Save images only at the final epoch
        is_final_epoch = (epoch + 1 == config.NUM_EPOCHS)
        if is_final_epoch:
            print(f"📸 Saving final images...")
            
            # Save 10 reconstructed images at the final epoch
            save_reconstructed_images(
                generator, 
                dataloader, 
                device, 
                reconstructed_dir, 
                epoch + 1, 
                num_samples=10
            )
            
            # Save sample comparisons
            display_sample_images(
                generator, 
                dataloader, 
                device, 
                samples_dir,
                epoch + 1, 
                num_samples=min(3, config.NUM_DISPLAY_SAMPLES)
            )
        
        # Save checkpoint for every epoch
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'g_losses': g_losses,
            'd_losses': d_losses
        }, checkpoint_path)
        print(f"💾 Checkpoint saved at {checkpoint_path}")
        
        # Save intermediate checkpoint at specified intervals
        if (epoch + 1) % config.SAVE_INTERVAL == 0:
            intermediate_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'g_losses': g_losses,
                'd_losses': d_losses
            }, intermediate_checkpoint_path)
            print(f"💾 Saved intermediate checkpoint at epoch {epoch+1}")

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
    
    return fixed_results_dir  # Return the output directory for reference

if __name__ == "__main__":
    from configs.config import Config
    config = Config()
    train_stylegan3(config)
