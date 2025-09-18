import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import cv2
from PIL import Image
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EEGElectrodeMapping:
    """Comprehensive mapping of EEG electrodes to brain regions"""
    
    def __init__(self):
        # Standard 14-electrode EEG system mapping
        self.electrode_positions = {
            0: {'name': 'Fp1', 'region': 'Frontal', 'hemisphere': 'Left', 'area': 'Prefrontal'},
            1: {'name': 'Fp2', 'region': 'Frontal', 'hemisphere': 'Right', 'area': 'Prefrontal'},
            2: {'name': 'F3', 'region': 'Frontal', 'hemisphere': 'Left', 'area': 'Motor'},
            3: {'name': 'F4', 'region': 'Frontal', 'hemisphere': 'Right', 'area': 'Motor'},
            4: {'name': 'F7', 'region': 'Frontal', 'hemisphere': 'Left', 'area': 'Temporal-Frontal'},
            5: {'name': 'F8', 'region': 'Frontal', 'hemisphere': 'Right', 'area': 'Temporal-Frontal'},
            6: {'name': 'C3', 'region': 'Central', 'hemisphere': 'Left', 'area': 'Sensorimotor'},
            7: {'name': 'C4', 'region': 'Central', 'hemisphere': 'Right', 'area': 'Sensorimotor'},
            8: {'name': 'T5', 'region': 'Temporal', 'hemisphere': 'Left', 'area': 'Posterior-Temporal'},
            9: {'name': 'T6', 'region': 'Temporal', 'hemisphere': 'Right', 'area': 'Posterior-Temporal'},
            10: {'name': 'P3', 'region': 'Parietal', 'hemisphere': 'Left', 'area': 'Somatosensory'},
            11: {'name': 'P4', 'region': 'Parietal', 'hemisphere': 'Right', 'area': 'Somatosensory'},
            12: {'name': 'O1', 'region': 'Occipital', 'hemisphere': 'Left', 'area': 'Visual'},
            13: {'name': 'O2', 'region': 'Occipital', 'hemisphere': 'Right', 'area': 'Visual'}
        }
        
        self.region_to_id = {'Frontal': 0, 'Central': 1, 'Temporal': 2, 'Parietal': 3, 'Occipital': 4}
        self.hemisphere_to_id = {'Left': 0, 'Right': 1}
        
        self.functional_areas = {
            'Prefrontal': {'id': 0, 'function': 'Executive Control, Decision Making'},
            'Motor': {'id': 1, 'function': 'Motor Control, Movement Planning'},
            'Temporal-Frontal': {'id': 2, 'function': 'Language, Working Memory'},
            'Sensorimotor': {'id': 3, 'function': 'Touch, Motor Execution'},
            'Posterior-Temporal': {'id': 4, 'function': 'Auditory Processing, Memory'},
            'Somatosensory': {'id': 5, 'function': 'Touch, Spatial Processing'},
            'Visual': {'id': 6, 'function': 'Visual Processing'}
        }
    
    def get_electrode_info(self, electrode_idx):
        """Get comprehensive electrode information"""
        if electrode_idx in self.electrode_positions:
            info = self.electrode_positions[electrode_idx].copy()
            info['region_id'] = self.region_to_id[info['region']]
            info['hemisphere_id'] = self.hemisphere_to_id[info['hemisphere']]
            info['functional_area_id'] = self.functional_areas[info['area']]['id']
            info['function'] = self.functional_areas[info['area']]['function']
            return info
        return None

class HeatmapProcessor:
    """Process heatmap images and extract connectivity patterns"""
    
    @staticmethod
    def load_heatmap_image(image_path):
        """Load and preprocess heatmap image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
            image = image.astype(np.float32) / 255.0
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    @staticmethod
    def extract_connectivity_matrix_from_heatmap(image_path, matrix_size=14):
        """Extract connectivity matrix from heatmap image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            matrix_region = cv2.resize(gray, (matrix_size, matrix_size))
            connectivity_matrix = matrix_region.astype(np.float32) / 255.0
            
            return connectivity_matrix
        except Exception as e:
            print(f"Error extracting matrix from {image_path}: {e}")
            return None
    
    @staticmethod
    def detect_low_activity_nodes(connectivity_matrix, threshold_percentile=25):
        """Detect low activity nodes from connectivity matrix"""
        if connectivity_matrix is None:
            return [], np.array([])
        
        node_activity = np.sum(connectivity_matrix, axis=1)
        threshold = np.percentile(node_activity, threshold_percentile)
        low_activity_nodes = np.where(node_activity <= threshold)[0]
        
        # Ensure reasonable number of nodes
        if len(low_activity_nodes) == 0:
            low_activity_nodes = [np.argmin(node_activity)]
        elif len(low_activity_nodes) > 8:
            sorted_indices = np.argsort(node_activity)
            low_activity_nodes = sorted_indices[:8]
        
        return low_activity_nodes.tolist(), node_activity

class BrainNodeDataset(Dataset):
    """Dataset for brain node classification from heatmaps"""
    
    def __init__(self, heatmap_dir, transform=None):
        self.heatmap_dir = heatmap_dir
        self.transform = transform
        self.samples = []
        self.electrode_mapping = EEGElectrodeMapping()
        
        self._load_samples()
    
    def _load_samples(self):
        """Load heatmap samples and extract node information"""
        if not os.path.exists(self.heatmap_dir):
            print(f"Heatmap directory not found: {self.heatmap_dir}")
            return
        
        heatmap_files = [f for f in os.listdir(self.heatmap_dir) if f.endswith('.png')]
        print(f"Found {len(heatmap_files)} heatmap files")
        
        for filename in heatmap_files:
            try:
                parts = filename.split('_')
                severity = int(parts[1].replace('sev', ''))
                
                filepath = os.path.join(self.heatmap_dir, filename)
                
                image = HeatmapProcessor.load_heatmap_image(filepath)
                if image is None:
                    continue
                
                connectivity_matrix = HeatmapProcessor.extract_connectivity_matrix_from_heatmap(filepath)
                if connectivity_matrix is None:
                    continue
                
                low_activity_nodes, node_activity = HeatmapProcessor.detect_low_activity_nodes(connectivity_matrix)
                
                for node_idx in low_activity_nodes:
                    electrode_info = self.electrode_mapping.get_electrode_info(node_idx)
                    if electrode_info:
                        sample = {
                            'image': image,
                            'connectivity_matrix': connectivity_matrix,
                            'node_idx': node_idx,
                            'severity': severity,
                            'region_id': electrode_info['region_id'],
                            'hemisphere_id': electrode_info['hemisphere_id'],
                            'functional_area_id': electrode_info['functional_area_id'],
                            'electrode_name': electrode_info['name'],
                            'node_activity': node_activity[node_idx],
                            'filename': filename
                        }
                        self.samples.append(sample)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        print(f"Created {len(self.samples)} training samples from low activity nodes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = torch.FloatTensor(sample['image']).permute(2, 0, 1)
        connectivity_matrix = torch.FloatTensor(sample['connectivity_matrix']).unsqueeze(0)
        
        connectivity_resized = F.interpolate(
            connectivity_matrix.unsqueeze(0),
            size=(224, 224), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
        
        combined_input = torch.cat([image, connectivity_resized], dim=0)
        
        return {
            'input': combined_input,
            'node_idx': torch.tensor(sample['node_idx'], dtype=torch.long),
            'severity': torch.tensor(sample['severity'], dtype=torch.long),
            'region_id': torch.tensor(sample['region_id'], dtype=torch.long),
            'hemisphere_id': torch.tensor(sample['hemisphere_id'], dtype=torch.long),
            'functional_area_id': torch.tensor(sample['functional_area_id'], dtype=torch.long),
            'node_activity': torch.tensor(sample['node_activity'], dtype=torch.float32),
            'electrode_name': sample['electrode_name']
        }

class BrainNodeClassifierCNN(nn.Module):
    """3-Layer CNN for brain node classification from heatmaps"""
    
    def __init__(self, input_channels=4, num_electrodes=14, num_regions=5, 
                 num_hemispheres=2, num_functional_areas=7):
        super(BrainNodeClassifierCNN, self).__init__()
        
        self.input_channels = input_channels
        
        # Layer 1: Feature extraction
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Layer 2: Enhanced feature learning
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Layer 3: High-level feature abstraction
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc_input_size = 256 * 4 * 4
        
        self.shared_fc = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Task-specific heads
        self.electrode_head = nn.Linear(256, num_electrodes)
        self.region_head = nn.Linear(256, num_regions)
        self.hemisphere_head = nn.Linear(256, num_hemispheres)
        self.functional_head = nn.Linear(256, num_functional_areas)
        
        self.activity_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        shared_features = self.shared_fc(x)
        
        return {
            'electrode': self.electrode_head(shared_features),
            'region': self.region_head(shared_features),
            'hemisphere': self.hemisphere_head(shared_features),
            'functional_area': self.functional_head(shared_features),
            'activity': self.activity_head(shared_features),
            'features': shared_features
        }

class BrainNodeTrainer:
    """Trainer for the brain node classifier"""
    
    def __init__(self, model, device, learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.7)
        
        self.classification_criterion = nn.CrossEntropyLoss()
        self.regression_criterion = nn.MSELoss()
        
        self.loss_weights = {
            'electrode': 2.0,
            'region': 1.5,
            'hemisphere': 1.0,
            'functional_area': 1.2,
            'activity': 0.8
        }
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct_electrode = 0
        correct_region = 0
        total_samples = 0
        
        for batch in dataloader:
            inputs = batch['input'].to(self.device)
            electrode_targets = batch['node_idx'].to(self.device)
            region_targets = batch['region_id'].to(self.device)
            hemisphere_targets = batch['hemisphere_id'].to(self.device)
            functional_targets = batch['functional_area_id'].to(self.device)
            activity_targets = batch['node_activity'].to(self.device).unsqueeze(1)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            
            electrode_loss = self.classification_criterion(outputs['electrode'], electrode_targets)
            region_loss = self.classification_criterion(outputs['region'], region_targets)
            hemisphere_loss = self.classification_criterion(outputs['hemisphere'], hemisphere_targets)
            functional_loss = self.classification_criterion(outputs['functional_area'], functional_targets)
            activity_loss = self.regression_criterion(outputs['activity'], activity_targets)
            
            total_batch_loss = (
                self.loss_weights['electrode'] * electrode_loss +
                self.loss_weights['region'] * region_loss +
                self.loss_weights['hemisphere'] * hemisphere_loss +
                self.loss_weights['functional_area'] * functional_loss +
                self.loss_weights['activity'] * activity_loss
            )
            
            total_batch_loss.backward()
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            correct_electrode += (outputs['electrode'].argmax(1) == electrode_targets).sum().item()
            correct_region += (outputs['region'].argmax(1) == region_targets).sum().item()
            total_samples += inputs.size(0)
        
        avg_loss = total_loss / len(dataloader)
        electrode_acc = correct_electrode / total_samples
        region_acc = correct_region / total_samples
        
        return avg_loss, electrode_acc, region_acc
    
    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct_electrode = 0
        correct_region = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input'].to(self.device)
                electrode_targets = batch['node_idx'].to(self.device)
                region_targets = batch['region_id'].to(self.device)
                hemisphere_targets = batch['hemisphere_id'].to(self.device)
                functional_targets = batch['functional_area_id'].to(self.device)
                activity_targets = batch['node_activity'].to(self.device).unsqueeze(1)
                
                outputs = self.model(inputs)
                
                electrode_loss = self.classification_criterion(outputs['electrode'], electrode_targets)
                region_loss = self.classification_criterion(outputs['region'], region_targets)
                hemisphere_loss = self.classification_criterion(outputs['hemisphere'], hemisphere_targets)
                functional_loss = self.classification_criterion(outputs['functional_area'], functional_targets)
                activity_loss = self.regression_criterion(outputs['activity'], activity_targets)
                
                total_batch_loss = (
                    self.loss_weights['electrode'] * electrode_loss +
                    self.loss_weights['region'] * region_loss +
                    self.loss_weights['hemisphere'] * hemisphere_loss +
                    self.loss_weights['functional_area'] * functional_loss +
                    self.loss_weights['activity'] * activity_loss
                )
                
                total_loss += total_batch_loss.item()
                correct_electrode += (outputs['electrode'].argmax(1) == electrode_targets).sum().item()
                correct_region += (outputs['region'].argmax(1) == region_targets).sum().item()
                total_samples += inputs.size(0)
        
        avg_loss = total_loss / len(dataloader)
        electrode_acc = correct_electrode / total_samples
        region_acc = correct_region / total_samples
        
        return avg_loss, electrode_acc, region_acc
    
    def train(self, train_loader, val_loader, num_epochs=50):
        """Train the model"""
        print(f"🚀 Starting training for {num_epochs} epochs")
        print(f"📊 Training samples: {len(train_loader.dataset)}")
        print(f"📊 Validation samples: {len(val_loader.dataset)}")
        
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            train_loss, train_electrode_acc, train_region_acc = self.train_epoch(train_loader)
            val_loss, val_electrode_acc, val_region_acc = self.validate(val_loader)
            
            self.scheduler.step()
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_electrode_acc)
            self.val_accuracies.append(val_electrode_acc)
            
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                print(f"Epoch [{epoch:3d}/{num_epochs}] "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Train Electrode Acc: {train_electrode_acc:.4f}, "
                      f"Val Electrode Acc: {val_electrode_acc:.4f}, "
                      f"Val Region Acc: {val_region_acc:.4f}")
            
            if val_electrode_acc > best_val_acc:
                best_val_acc = val_electrode_acc
                torch.save(self.model.state_dict(), 'best_brain_node_classifier.pth')
                print(f"💾 New best model saved! Validation accuracy: {best_val_acc:.4f}")
        
        print(f"✅ Training completed! Best validation accuracy: {best_val_acc:.4f}")
        
        # Plot training history
        self._plot_training_history()
        
        return self.train_losses, self.val_losses
    
    def _plot_training_history(self):
        """Plot training and validation curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Training Loss', color='blue')
        ax1.plot(self.val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accuracies, label='Training Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Training history plot saved as 'training_history.png'")

def create_demo_heatmaps(output_dir, num_samples=20):
    """Create demo heatmaps for testing if no real data available"""
    print(f"🎨 Creating demo heatmaps in: {output_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for sev in range(5):  # 5 severity levels
        for sample in range(num_samples // 5):
            heatmap = np.random.rand(224, 224, 3) * 255
            
            if sev > 0:
                mask = np.random.rand(224, 224) < (sev * 0.15)
                heatmap[mask] = heatmap[mask] * 0.3
            
            filename = f"heatmap_sev{sev}_sample{sample}.png"
            filepath = os.path.join(output_dir, filename)
            
            heatmap_uint8 = heatmap.astype(np.uint8)
            cv2.imwrite(filepath, cv2.cvtColor(heatmap_uint8, cv2.COLOR_RGB2BGR))
    
    print(f"✅ Created {num_samples} demo heatmaps")

def main():
    """Main training function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Configuration
    HEATMAP_DIR = "E:/Desktop/IEEE/heatmap_output"
    EPOCHS = 50
    LEARNING_RATE = 0.001
    BATCH_SIZE = 8
    
    print("🧠 BRAIN NODE CLASSIFICATION - TRAINING MODULE")
    print("=" * 60)
    print("🎯 Training Objectives:")
    print("   • Learn to identify electrode positions from connectivity patterns")
    print("   • Classify brain regions (Frontal, Central, Temporal, Parietal, Occipital)")
    print("   • Predict hemispheres and functional areas")
    print("   • Estimate neural activity levels")
    print("=" * 60)
    
    # Check if heatmap directory exists
    if not os.path.exists(HEATMAP_DIR):
        print(f"⚠️ Heatmap directory not found: {HEATMAP_DIR}")
        print("🎨 Creating demo heatmaps for training...")
        create_demo_heatmaps(HEATMAP_DIR, 30)
    
    # Create dataset
    print("\n🔄 Loading training dataset...")
    dataset = BrainNodeDataset(HEATMAP_DIR)
    
    if len(dataset) == 0:
        print("❌ No training data found! Creating demo data...")
        create_demo_heatmaps(HEATMAP_DIR, 30)
        dataset = BrainNodeDataset(HEATMAP_DIR)
    
    if len(dataset) == 0:
        print("❌ Still no data found! Please check the directory and file format.")
        return
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"📊 Dataset split: {train_size} train, {val_size} validation")
    
    # Create model
    print("\n🏗️ Building 3-Layer CNN model...")
    model = BrainNodeClassifierCNN(
        input_channels=4,  # RGB + connectivity matrix
        num_electrodes=14,
        num_regions=5,
        num_hemispheres=2,
        num_functional_areas=7
    )
    
    print(f"🔧 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n🚀 Starting training process...")
    trainer = BrainNodeTrainer(model, device, learning_rate=LEARNING_RATE)
    train_losses, val_losses = trainer.train(train_loader, val_loader, num_epochs=EPOCHS)
    
    # Save final model info
    model_info = {
        'model_architecture': 'BrainNodeClassifierCNN',
        'input_channels': 4,
        'num_electrodes': 14,
        'num_regions': 5,
        'num_hemispheres': 2,
        'num_functional_areas': 7,
        'training_epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'training_samples': train_size,
        'validation_samples': val_size,
        'device_used': str(device),
        'training_completed': datetime.now().isoformat()
    }
    
    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("\n✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("📁 Generated Files:")
    print("   • best_brain_node_classifier.pth - Trained model weights")
    print("   • model_info.json - Model configuration and training info")
    print("   • training_history.png - Training curves")
    print()
    print("🔄 Next Steps:")
    print("   1. Use 'braincnn_test.py' to test individual heatmaps")
    print("   2. The trained model is ready for brain area prediction")
    print("   3. Check training_history.png to evaluate model performance")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Training Error: {e}")
        import traceback
        traceback.print_exc()
