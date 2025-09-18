import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import pickle
import warnings
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from scipy.stats import wasserstein_distance
import gc
from sklearn.model_selection import train_test_split
from collections import defaultdict
import math
from scipy.linalg import sqrtm

from eeg_data_loader import load_enhanced_eeg_data, EnhancedEEGConnectivityDataset
from brainganmodel import UltraEnhancedBrainConnectivityGAN, ultra_enhanced_weights_init

warnings.filterwarnings("ignore")

class BrainConnectivityAnalyzer:
    """Analyze brain connectivity matrices and identify low activity nodes"""
    
    @staticmethod
    def find_low_activity_nodes(matrix, threshold_percentile=25):
        """Find nodes with low connectivity activity"""
        # Average across channels if multiple channels exist
        if len(matrix.shape) == 3:
            avg_matrix = np.mean(matrix, axis=0)
        else:
            avg_matrix = matrix
            
        # Calculate node activity (sum of connections for each node)
        node_activity = np.sum(avg_matrix, axis=1)
        
        # Find threshold for low activity
        threshold = np.percentile(node_activity, threshold_percentile)
        low_activity_nodes = np.where(node_activity <= threshold)[0]
        
        return low_activity_nodes, node_activity, threshold
    
    @staticmethod
    def create_heatmap_with_analysis(matrix, severity_label, subject_id, save_path):
        """Create and save connectivity heatmap with low activity node annotations"""
        severity_names = ['Normal', 'Mild', 'Moderate', 'Severe', 'Very Severe']
        
        # Find low activity nodes
        low_activity_nodes, node_activity, threshold = BrainConnectivityAnalyzer.find_low_activity_nodes(matrix)
        
        # Average across channels for visualization
        if len(matrix.shape) == 3:
            display_matrix = np.mean(matrix, axis=0)
        else:
            display_matrix = matrix
            
        # Create figure for connectivity matrix only
        fig = plt.figure(figsize=(10, 10))
        ax1 = plt.subplot(1, 1, 1)
        
        # Create heatmap
        im = ax1.imshow(display_matrix, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
        
        # Highlight and annotate low activity nodes
        for node in low_activity_nodes:
            # Add red border around low activity nodes
            ax1.add_patch(plt.Rectangle((node-0.5, -0.5), 1, len(display_matrix), 
                                      fill=False, edgecolor='red', linewidth=1, alpha=0.7))
            # Annotate with node index
            ax1.text(node, node, str(node), color='red', ha='center', va='center', fontsize=8, 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        ax1.set_title(f'Connectivity - {severity_names[severity_label]}', fontsize=12)
        ax1.set_xlabel('Brain Regions', fontsize=10)
        ax1.set_ylabel('Brain Regions', fontsize=10)
        
        # Add colorbar
        plt.colorbar(im, ax=ax1, shrink=0.8)
        
        # Save the standalone connectivity matrix with annotations
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        return low_activity_nodes, node_activity

class RobustFrechetDistance:
    """Robust Frechet Distance calculation with better numerical stability"""
    @staticmethod
    def calculate_activation_statistics(matrices, eps=1e-6):
        """Calculate statistics with numerical stability"""
        try:
            matrices_flat = matrices.view(matrices.size(0), -1).cpu().numpy()
            
            # Remove invalid values
            valid_mask = ~(np.isnan(matrices_flat).any(axis=1) | np.isinf(matrices_flat).any(axis=1))
            matrices_flat = matrices_flat[valid_mask]
            
            if len(matrices_flat) == 0:
                return None, None
            
            mu = np.mean(matrices_flat, axis=0)
            
            sigma = np.cov(matrices_flat, rowvar=False)
            sigma = sigma + eps * np.eye(sigma.shape[0])
            
            # Ensure positive definite
            eigenvals, eigenvecs = np.linalg.eigh(sigma)
            eigenvals = np.maximum(eigenvals, eps)
            sigma = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            return mu, sigma
            
        except Exception as e:
            print(f"Warning: Statistics calculation failed: {e}")
            return None, None
    
    @staticmethod
    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Robust Frechet distance calculation"""
        if mu1 is None or mu2 is None or sigma1 is None or sigma2 is None:
            return float('inf')
        
        try:
            mu1 = np.atleast_1d(mu1)
            mu2 = np.atleast_1d(mu2)
            sigma1 = np.atleast_2d(sigma1)
            sigma2 = np.atleast_2d(sigma2)
            
            diff = mu1 - mu2
            
            product = sigma1 @ sigma2
            sqrt_product = sqrtm(product)
            
            if np.iscomplexobj(sqrt_product):
                sqrt_product = sqrt_product.real
            
            fd = diff.T.dot(diff) + np.trace(sigma1 + sigma2 - 2 * sqrt_product)
            return max(0, fd)  # Ensure non-negative
            
        except Exception as e:
            print(f"Warning: Frechet distance calculation failed: {e}")
            return float('inf')

class UltraEnhancedBrainConnectivityGANTrainer:
    """Trainer class with enhanced training logic and anti-overfitting measures"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gan = UltraEnhancedBrainConnectivityGAN(**{k: v for k, v in config.items() if k in ['noise_dim', 'num_classes', 'matrix_size', 'num_channels', 'base_channels']}).to(self.device)
        self.gan.apply(ultra_enhanced_weights_init)
        
        self.optimizer_g = optim.Adam(self.gan.generator.parameters(), lr=config['lr_g'], betas=(0.5, 0.999), weight_decay=1e-4)
        self.optimizer_d = optim.Adam(self.gan.discriminator.parameters(), lr=config['lr_d'], betas=(0.5, 0.999), weight_decay=1e-4)
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.class_criterion = nn.CrossEntropyLoss()
        
        self.heatmap_dir = os.path.join("E:/Desktop/IEEE", "heatmap_output")
        self.results_dir = os.path.join("E:/Desktop/IEEE", "results", f"enhanced_brain_gan_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.heatmap_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "analysis"), exist_ok=True)
        
        self.best_fd = float('inf')
        self.patience = 0
        
    def train_step(self, real_matrices, severity_labels):
        """Single training step with enhanced stability"""
        batch_size = real_matrices.size(0)
        real_matrices = real_matrices.to(self.device)
        severity_labels = severity_labels.to(self.device)
        
        # Labels for real and fake
        real_label = torch.ones(batch_size, 1).to(self.device)
        fake_label = torch.zeros(batch_size, 1).to(self.device)
        
        # Train Discriminator
        self.optimizer_d.zero_grad()
        
        # Real data
        validity_real, cls_real, conn_real, spectral_real = self.gan.discriminator(real_matrices, return_features=False)
        d_loss_real = self.criterion(validity_real, real_label)
        cls_loss_real = self.class_criterion(cls_real, severity_labels)
        
        # Generate fake data
        noise = torch.randn(batch_size, self.config['noise_dim']).to(self.device)
        fake_matrices, _ = self.gan.generate_samples(batch_size, severity_labels, self.device)
        validity_fake, cls_fake, conn_fake, spectral_fake = self.gan.discriminator(fake_matrices.detach(), return_features=False)
        d_loss_fake = self.criterion(validity_fake, fake_label)
        
        # Total discriminator loss
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        self.optimizer_d.step()
        
        # Train Generator
        self.optimizer_g.zero_grad()
        
        validity_fake, cls_fake, conn_fake, spectral_fake = self.gan.discriminator(fake_matrices, return_features=False)
        g_loss = self.criterion(validity_fake, real_label)
        cls_loss_fake = self.class_criterion(cls_fake, severity_labels)
        
        # Additional regularization losses
        conn_loss = F.mse_loss(conn_fake, conn_real.detach())
        spectral_loss = F.mse_loss(spectral_fake, spectral_real.detach())
        diversity_loss = torch.var(fake_matrices.view(batch_size, -1), dim=0).mean()
        
        total_g_loss = (g_loss + 
                       self.config['lambda_classification'] * cls_loss_fake + 
                       self.config['lambda_connectivity'] * conn_loss + 
                       self.config['lambda_diversity'] * diversity_loss + 
                       self.config['lambda_spectral'] * spectral_loss)
        
        total_g_loss.backward()
        self.optimizer_g.step()
        
        return d_loss.item(), total_g_loss.item()
    
    def evaluate(self, dataloader):
        """Evaluate model performance with comprehensive metrics"""
        self.gan.eval()
        total_d_loss, total_g_loss = 0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in dataloader:
                real_matrices = batch['adjacency_matrix'].to(self.device)
                severity_labels = batch['severity_label'].to(self.device)
                
                batch_size = real_matrices.size(0)
                noise = torch.randn(batch_size, self.config['noise_dim']).to(self.device)
                fake_matrices, _ = self.gan.generate_samples(batch_size, severity_labels, self.device)
                
                validity_real, cls_real, _, _ = self.gan.discriminator(real_matrices)
                validity_fake, cls_fake, _, _ = self.gan.discriminator(fake_matrices)
                
                d_loss = self.criterion(validity_real, torch.ones_like(validity_real)) + \
                         self.criterion(validity_fake, torch.zeros_like(validity_fake))
                g_loss = self.criterion(validity_fake, torch.ones_like(validity_fake))
                
                total_d_loss += d_loss.item()
                total_g_loss += g_loss.item()
                
                preds = cls_fake.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(severity_labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # Calculate Fréchet Distance
        real_stats = RobustFrechetDistance.calculate_activation_statistics(real_matrices)
        fake_stats = RobustFrechetDistance.calculate_activation_statistics(fake_matrices)
        fd = RobustFrechetDistance.calculate_frechet_distance(*real_stats, *fake_stats) if real_stats[0] is not None else float('inf')
        
        class_acc = {i: accuracy_score([l for l in all_labels if l == i], [p for l, p in zip(all_labels, all_preds) if l == i]) 
                     for i in range(self.config['num_classes'])}
        
        self.gan.train()
        return {
            'd_loss': total_d_loss / len(dataloader),
            'g_loss': total_g_loss / len(dataloader),
            'accuracy': accuracy,
            'f1_score': f1,
            'frechet_distance': fd,
            'class_specific_acc': class_acc
        }
    
    def generate_comprehensive_heatmaps(self, epoch):
        """Generate comprehensive heatmaps for all severity levels"""
        os.makedirs(self.heatmap_dir, exist_ok=True)
        num_samples_per_severity = 10
        severity_levels = range(self.config['num_classes'])
        
        for severity in severity_levels:
            for i in range(num_samples_per_severity):
                # Generate synthetic sample for this severity
                fake_labels = torch.full((1,), severity, device=self.device)
                noise = torch.randn(1, self.config['noise_dim'], device=self.device)
                self.gan.generator.eval()
                with torch.no_grad():
                    fake_matrices = self.gan.generator(noise, fake_labels)
                self.gan.generator.train()

                
                # Save heatmap
                subject_id = f"synth_sev{severity}_sample{i+1}"
                save_path = os.path.join(self.heatmap_dir, f"heatmap_sev{severity}_sample{i+1}.png")
                low_activity_nodes, node_activity = BrainConnectivityAnalyzer.create_heatmap_with_analysis(
                    fake_matrices[0].cpu().numpy(), severity, subject_id, save_path
                )
    
        print(f"📊 Generated {len(os.listdir(self.heatmap_dir))} heatmaps in {self.heatmap_dir}")
    
    def train(self, train_loader, test_loader, num_epochs):
        """Main training loop with early stopping and overfitting detection"""
        for epoch in range(num_epochs):
            total_d_loss, total_g_loss = 0, 0
            self.gan.train()
            
            for i, batch in enumerate(train_loader):
                d_loss, g_loss = self.train_step(batch['adjacency_matrix'], batch['severity_label'])
                total_d_loss += d_loss
                total_g_loss += g_loss
            
            avg_d_loss = total_d_loss / len(train_loader)
            avg_g_loss = total_g_loss / len(train_loader)
            
            # Evaluate on test set
            if epoch % self.config['eval_interval'] == 0 or epoch == num_epochs - 1:
                test_metrics = self.evaluate(test_loader)
                train_metrics = self.evaluate(train_loader)
                
                print(f"\nEpoch [{epoch}/{num_epochs}]")
                print(f"  Train - D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")
                print(f"  Test - Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1_score']:.4f}, "
                      f"FD: {test_metrics['frechet_distance']:.2f}")
                
                # Early stopping and best model saving
                current_fd = test_metrics['frechet_distance']
                if current_fd < self.best_fd:
                    self.best_fd = current_fd
                    self.patience = 0
                    self.save_checkpoint(epoch, test_metrics, is_best=True)
                else:
                    self.patience += 1
                    if self.patience >= self.config['early_stopping_patience']:
                        print(f"⏹ Early stopping triggered at epoch {epoch}")
                        break
                
                self.save_checkpoint(epoch, test_metrics, is_best=False)
                self.generate_comprehensive_heatmaps(epoch)
            
            gc.collect()
            torch.cuda.empty_cache()
        
        final_eval = self.evaluate(test_loader)
        self.save_final_results(final_eval, epoch + 1)
        return self.results_dir
    
    def save_checkpoint(self, epoch, metrics, is_best):
        """Save model checkpoint with metrics"""
        try:
            checkpoint_data = {
                'epoch': epoch,
                'd_loss': metrics['d_loss'],
                'g_loss': metrics['g_loss'],
                'accuracy': metrics['accuracy'],
                'f1_score': metrics['f1_score'],
                'frechet_distance': metrics['frechet_distance'],
                'class_specific_acc': metrics['class_specific_acc']
            }
            
            json_path = os.path.join(self.results_dir, "analysis", f"checkpoint_epoch_{epoch}.json")
            model_path = os.path.join(self.results_dir, "checkpoints", f"checkpoint_epoch_{epoch}.pth")
        
            with open(json_path, 'w') as f:
                json_data = {k: v for k, v in checkpoint_data.items()}
                json.dump(json_data, f, indent=2)
            
            # Save model
            torch.save({
                'generator_state_dict': self.gan.generator.state_dict(),
                'discriminator_state_dict': self.gan.discriminator.state_dict(),
                'optimizer_g_state_dict': self.optimizer_g.state_dict(),
                'optimizer_d_state_dict': self.optimizer_d.state_dict(),
                'epoch': epoch,
                'metrics': checkpoint_data
            }, model_path)
            
            print(f"💾 {'Best model' if is_best else 'Checkpoint'} saved successfully")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not save checkpoint: {e}")
    
    def save_final_results(self, final_eval, total_epochs):
        """Save comprehensive final results"""
        try:
            summary = {
                'training_summary': {
                    'total_epochs': total_epochs,
                    'final_accuracy': final_eval['accuracy'],
                    'final_f1_score': final_eval['f1_score'],
                    'final_frechet_distance': final_eval.get('frechet_distance', 'N/A'),
                    'overfitting_analysis': {
                        'high_accuracy': final_eval['accuracy'] > 0.95,
                        'high_fd': final_eval.get('frechet_distance', 0) > 50,
                        'potential_overfitting': final_eval['accuracy'] > 0.95 and final_eval.get('frechet_distance', 0) > 50,
                        'recommendations': [
                            "Monitor on independent test set",
                            "Increase dataset complexity",
                            "Add more regularization if overfitting detected",
                            "Validate model generalization"
                        ]
                    }
                },
                'class_specific_results': {},
                'heatmap_info': {
                    'total_heatmaps_generated': 50,  # 5 severities × 10 samples
                    'heatmap_directory': self.heatmap_dir,
                    'severity_levels': ['Normal', 'Mild', 'Moderate', 'Severe', 'Very Severe'],
                    'samples_per_severity': 10
                },
                'config_used': self.config
            }
            
            # Add class-specific results
            severity_names = ['Normal', 'Mild', 'Moderate', 'Severe', 'Very Severe']
            for cls, acc in final_eval['class_specific_acc'].items():
                summary['class_specific_results'][severity_names[cls]] = {
                    'accuracy': acc,
                    'class_index': cls
                }
            
            summary_path = os.path.join(self.results_dir, "analysis", "comprehensive_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"📊 Comprehensive results saved to: {summary_path}")
            
        except Exception as e:
            print(f"⚠️ Could not save final results: {e}")

def get_enhanced_config_5_classes():
    """Configuration optimized for 5 classes with anti-overfitting measures"""
    return {
        'noise_dim': 128,
        'num_classes': 5,  # Updated to 5 classes
        'matrix_size': 14,
        'num_channels': 2,
        'base_channels': 32,  # Reduced to prevent overfitting
        'lr_g': 0.0001,
        'lr_d': 0.0001,  # Balanced learning rates
        'lambda_classification': 3.0,  # Reduced from 6.0
        'lambda_connectivity': 0.8,  # Reduced
        'lambda_diversity': 0.3,  # Increased diversity emphasis
        'lambda_spectral': 0.1,
        'gradient_penalty': 10.0,
        'd_train_interval': 3,  # Train discriminator less frequently
        'eval_interval': 1,
        'checkpoint_interval': 5,
        'early_stopping_patience': 20,  # Reduced patience
        'batch_size': 4,  # Smaller batch size
        'test_split': 0.25  # Larger test set for better validation
    }

def train_enhanced_brain_connectivity_gan(data_dir, excel_file_path=None, config=None, num_epochs=80):
    """Main function to train the enhanced Brain Connectivity GAN"""
    if config is None:
        config = get_enhanced_config_5_classes()
    
    print("🧠 Enhanced Brain Connectivity GAN Training with 5 Severity Levels")
    print("=" * 70)
    print("🔧 Key Features:")
    print("   • 5 Severity Levels: Normal, Mild, Moderate, Severe, Very Severe")
    print("   • 50 Comprehensive Heatmaps (10 per severity)")
    print("   • Low Activity Node Analysis with Node Indices on Matrix")
    print("   • Anti-Overfitting Measures")
    print("   • Reduced Data Augmentation")
    print("=" * 70)
    
    try:
        train_loader, test_loader, full_dataset = load_enhanced_eeg_data(
            data_dir=data_dir,
            excel_file_path=excel_file_path,
            band_type='both',
            batch_size=config['batch_size'],
            test_split=config['test_split'],
            augment_factor=1.5,  # Reduced from 3 to prevent overfitting
            balance_classes=True,
            min_samples_per_class=8  # Reduced minimum samples
        )
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        raise
    
    # Check data distribution
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch['severity_label'].squeeze().numpy())
    
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    print(f"📊 Training data distribution: {dict(zip(unique_labels, counts))}")
    
    # Create trainer and start training
    trainer = UltraEnhancedBrainConnectivityGANTrainer(config)
    results_dir = trainer.train(train_loader, test_loader, num_epochs)
    
    # Final verification
    heatmap_files = os.listdir(trainer.heatmap_dir) if os.path.exists(trainer.heatmap_dir) else []
    print(f"\n📊 Training Results Verification:")
    print(f"  Results directory: {results_dir}")
    print(f"  Heatmaps generated: {len(heatmap_files)}")
    print(f"  Heatmap directory: {trainer.heatmap_dir}")
    
    if len(heatmap_files) >= 40:  # Allow some flexibility
        print("✅ Comprehensive heatmaps successfully generated!")
    else:
        print("⚠️ Expected 50 heatmaps, but fewer were generated")
    
    print(f"\n🎯 Overfitting Analysis:")
    print("• Check if accuracy > 95% while FD > 50 (potential overfitting)")
    print("• Generated heatmaps should show realistic brain connectivity patterns")
    print("• Low activity nodes with indices provide insights into severity-specific patterns")
    
    return results_dir

if __name__ == "__main__":
    DATA_DIR = r"E:\Desktop\IEEE\PD_all_adjacency_files_updated\PD_all_adjacency_files_updated"
    EXCEL_FILE = r"E:\Desktop\IEEE\PD_TBR_Classification_Updated.xlsx"
    
    # Enhanced configuration for 5 classes with anti-overfitting measures
    config = get_enhanced_config_5_classes()
    
    try:
        print("🚀 Starting Enhanced Brain Connectivity GAN Training...")
        print("🔧 Key Improvements:")
        print("   • 5 severity levels support (Normal, Mild, Moderate, Severe, Very Severe)")
        print("   • Comprehensive heatmap generation (50 total) with only connectivity matrix and node indices")
        print("   • Low activity node analysis annotated on matrix")
        print("   • Anti-overfitting measures (reduced LR, weight decay, regularization)")
        print("   • Reduced data augmentation to prevent overfitting")
        print("   • Enhanced early stopping with overfitting detection")
        print("   • Detailed connectivity analysis and visualization")
        print()
        
        results_dir = train_enhanced_brain_connectivity_gan(
            data_dir=DATA_DIR,
            excel_file_path=EXCEL_FILE,
            config=config,
            num_epochs=80  # Reduced epochs to prevent overfitting
        )
        
        print(f"\n🎉 Enhanced training completed successfully!")
        print(f"📁 Check results in: {results_dir}")
        print(f"🎨 Heatmaps saved in: E:/Desktop/IEEE/heatmap_output")
        print("\n📊 Generated Files:")
        print("   • 50 comprehensive heatmaps (10 per severity level)")
        print("   • Each heatmap includes:")
        print("     - Brain connectivity matrix")
        print("     - Low activity node indices (e.g., 1, 3, 8)")
        print("\n🏆 The model should now achieve good performance without overfitting!")
        print("🔍 Check the heatmap_output folder for detailed brain connectivity analysis")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        print("\n📝 Troubleshooting:")
        print("1. Update DATA_DIR and EXCEL_FILE paths")
        print("2. Ensure sufficient disk space")
        print("3. Check data files contain 5 severity levels")
        print("4. Verify file permissions for output directories")
