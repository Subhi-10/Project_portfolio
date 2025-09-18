import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE, ADASYN
import glob
import warnings
import random
import math
from scipy import ndimage
from scipy.ndimage import gaussian_filter
warnings.filterwarnings('ignore')

class AdvancedConnectivityAugmentation:
    """Advanced augmentation techniques for connectivity matrices"""
    
    @staticmethod
    def gaussian_noise(matrix, std_range=(0.01, 0.05)):
        """Add Gaussian noise with random standard deviation"""
        std = np.random.uniform(*std_range)
        noise = np.random.normal(0, std, matrix.shape)
        return np.clip(matrix + noise, 0, 1)
    
    @staticmethod
    def connectivity_dropout(matrix, dropout_range=(0.05, 0.15)):
        """Randomly dropout connections (set to lower values)"""
        dropout_rate = np.random.uniform(*dropout_range)
        mask = np.random.random(matrix.shape) > dropout_rate
        dropped = matrix * mask + matrix * (1 - mask) * 0.1
        return dropped
    
    @staticmethod
    def severity_specific_augmentation(matrix, severity_label):
        """Apply severity-specific augmentations"""
        if severity_label == 0:  # Normal/Mild - add subtle variations
            return AdvancedConnectivityAugmentation.gaussian_noise(matrix, (0.005, 0.02))
        elif severity_label == 1:  # Moderate - medium augmentation
            if np.random.random() > 0.5:
                return AdvancedConnectivityAugmentation.gaussian_noise(matrix, (0.01, 0.03))
            else:
                return AdvancedConnectivityAugmentation.connectivity_dropout(matrix, (0.05, 0.10))
        else:  # Severe - stronger augmentation
            aug_type = np.random.choice(['noise', 'dropout', 'combined'])
            if aug_type == 'noise':
                return AdvancedConnectivityAugmentation.gaussian_noise(matrix, (0.02, 0.05))
            elif aug_type == 'dropout':
                return AdvancedConnectivityAugmentation.connectivity_dropout(matrix, (0.10, 0.20))
            else:  # combined
                matrix = AdvancedConnectivityAugmentation.gaussian_noise(matrix, (0.01, 0.03))
                return AdvancedConnectivityAugmentation.connectivity_dropout(matrix, (0.05, 0.10))
    
    @staticmethod
    def smooth_augmentation(matrix, sigma_range=(0.5, 1.5)):
        """Apply Gaussian smoothing for regularization"""
        sigma = np.random.uniform(*sigma_range)
        smoothed = np.zeros_like(matrix)
        for c in range(matrix.shape[0]):  # For each channel
            smoothed[c] = gaussian_filter(matrix[c], sigma=sigma)
        return np.clip(smoothed, 0, 1)
    
    @staticmethod
    def regional_scaling(matrix, num_regions=3):
        """Scale different brain regions differently"""
        channels, height, width = matrix.shape
        region_size = height // num_regions
        
        augmented = matrix.copy()
        for c in range(channels):
            for region in range(num_regions):
                start_idx = region * region_size
                end_idx = min((region + 1) * region_size, height)
                scale_factor = np.random.uniform(0.8, 1.2)
                
                augmented[c, start_idx:end_idx, start_idx:end_idx] *= scale_factor
        
        return np.clip(augmented, 0, 1)

class EnhancedEEGConnectivityDataset(Dataset):
    """Enhanced Dataset with better data handling and augmentation"""
    
    def __init__(self, data_dir, excel_file_path=None, band_type='both', 
                 normalize=True, augment=False, augment_factor=2, 
                 balance_classes=True, min_samples_per_class=10):
        """
        Args:
            data_dir: Directory containing .npy and .csv files for each subject
            excel_file_path: Path to Excel file with severity scores
            band_type: 'beta', 'theta', or 'both'
            normalize: Whether to normalize adjacency matrices
            augment: Whether to apply data augmentation
            augment_factor: How many augmented samples per original sample
            balance_classes: Whether to balance classes using augmentation
            min_samples_per_class: Minimum samples per class after augmentation
        """
        self.data_dir = data_dir
        self.band_type = band_type
        self.normalize = normalize
        self.augment = augment
        self.augment_factor = augment_factor
        self.balance_classes = balance_classes
        self.min_samples_per_class = min_samples_per_class
        
        # Load and prepare data
        self.matrices, self.severity_labels, self.subject_ids = self._load_data(excel_file_path)
        
        # Balance classes if requested
        if self.balance_classes and len(self.matrices) > 0:
            self.matrices, self.severity_labels, self.subject_ids = self._balance_classes()
        
        # Apply data augmentation if requested
        if self.augment and len(self.matrices) > 0:
            self.matrices, self.severity_labels, self.subject_ids = self._augment_data()
        
        # Print dataset info
        self._print_dataset_info()
        
    def _load_data(self, excel_file_path):
        """Enhanced data loading with better error handling"""
        matrices = []
        severity_labels = []
        subject_ids = []
        
        # Load severity data from Excel
        severity_data = self._load_severity_data(excel_file_path)
        
        # Find all subject files
        subject_files = self._find_subject_files()
        
        print(f"📁 Found files for {len(subject_files)} subjects")
        
        # Load matrices for each subject
        for subject_id, files in subject_files.items():
            try:
                matrix_data = self._load_subject_matrices(subject_id, files)
                if matrix_data is not None:
                    matrices.append(matrix_data)
                    subject_ids.append(subject_id)
                    
                    # Get severity label with better fallback
                    severity = self._get_severity_label(subject_id, severity_data)
                    severity_labels.append(severity)
                    
            except Exception as e:
                print(f"⚠️ Error loading data for subject {subject_id}: {e}")
        
        if len(matrices) == 0:
            print("❌ No data loaded successfully!")
            return np.array([]), np.array([]), []
        
        matrices = np.array(matrices)
        severity_labels = np.array(severity_labels)
        
        # Normalize matrices if requested
        if self.normalize:
            matrices = self._normalize_matrices(matrices)
        
        # Convert severity labels to categories
        severity_labels = self._discretize_severity(severity_labels)
        
        return matrices, severity_labels, subject_ids
    
    def _load_severity_data(self, excel_file_path):
        """Enhanced severity data loading"""
        severity_data = {}
        
        if not excel_file_path or not os.path.exists(excel_file_path):
            print("⚠️ No Excel file provided or file doesn't exist. Using default severity assignment.")
            return severity_data
        
        try:
            df_severity = pd.read_excel(excel_file_path)
            print(f"📊 Loaded severity data with columns: {df_severity.columns.tolist()}")
            
            # Try to find severity and ID columns
            severity_col = self._find_column(df_severity, 
                ['severity', 'Severity', 'SEVERITY', 'PD_Severity', 'pd_severity', 
                 'Score', 'score', 'Predicted_Severity', 'Class', 'class'])
            
            id_col = self._find_column(df_severity,
                ['subject_id', 'Subject_ID', 'ID', 'id', 'Subject', 'subject', 'Name'])
            
            if severity_col and id_col:
                for _, row in df_severity.iterrows():
                    subject_key = str(row[id_col]).strip()
                    severity_value = row[severity_col]
                    
                    # Try multiple subject ID formats
                    possible_keys = [
                        subject_key,
                        f"subject_{subject_key.zfill(2)}",
                        f"sub_{subject_key.zfill(2)}",
                        f"subject{subject_key.zfill(2)}",
                        subject_key.replace('subject_', '').replace('sub_', ''),
                    ]
                    
                    for key in possible_keys:
                        severity_data[key] = severity_value
                        
                print(f"✅ Loaded severity data for {len(set(severity_data.values()))} unique subjects")
            else:
                print(f"⚠️ Could not find severity column: {severity_col} or ID column: {id_col}")
                
        except Exception as e:
            print(f"⚠️ Error loading Excel file: {e}")
        
        return severity_data
    
    def _find_column(self, df, possible_names):
        """Find column by possible names"""
        for name in possible_names:
            if name in df.columns:
                return name
        return None
    
    def _find_subject_files(self):
        """Enhanced subject file discovery"""
        if self.band_type == 'both':
            file_patterns = ['*beta*.npy', '*theta*.npy']
        elif self.band_type == 'beta':
            file_patterns = ['*beta*.npy']
        else:
            file_patterns = ['*theta*.npy']
        
        subject_files = {}
        
        for pattern in file_patterns:
            files = glob.glob(os.path.join(self.data_dir, pattern))
            print(f"🔍 Found {len(files)} files matching pattern '{pattern}'")
            
            for file_path in files:
                filename = os.path.basename(file_path)
                subject_id = self._extract_subject_id(filename)
                
                if subject_id not in subject_files:
                    subject_files[subject_id] = {}
                
                if 'beta' in filename.lower():
                    subject_files[subject_id]['beta'] = file_path
                elif 'theta' in filename.lower():
                    subject_files[subject_id]['theta'] = file_path
        
        return subject_files
    
    def _extract_subject_id(self, filename):
        """Enhanced subject ID extraction"""
        import re
        
        # Try multiple patterns
        patterns = [
            r'(?:subject|sub)[-_]?(\d+)',  # subject_01, sub01, etc.
            r'S(\d+)',  # S01, S1, etc.
            r'P(\d+)',  # P01, P1, etc. (Patient)
            r'(\d+)',   # Just numbers
        ]
        
        filename_lower = filename.lower()
        
        for pattern in patterns:
            match = re.search(pattern, filename_lower)
            if match:
                subject_num = match.group(1).zfill(2)
                return f"subject_{subject_num}"
        
        # Fallback: use filename without extension
        return os.path.splitext(filename)[0]
    
    def _load_subject_matrices(self, subject_id, files):
        """Load matrices for a single subject"""
        if self.band_type == 'both':
            if 'beta' in files and 'theta' in files:
                beta_matrix = np.load(files['beta'])
                theta_matrix = np.load(files['theta'])
                
                # Validate matrices
                if beta_matrix.shape != theta_matrix.shape or len(beta_matrix.shape) != 2:
                    print(f"⚠️ Skipping {subject_id}: Matrix shape mismatch or not 2D")
                    return None
                
                # Stack as channels [channels, height, width]
                combined_matrix = np.stack([beta_matrix, theta_matrix], axis=0)
                return combined_matrix
            else:
                print(f"⚠️ Skipping {subject_id}: Missing beta or theta file")
                return None
                
        elif self.band_type in ['beta', 'theta']:
            if self.band_type in files:
                matrix = np.load(files[self.band_type])
                if len(matrix.shape) != 2:
                    print(f"⚠️ Skipping {subject_id}: Matrix not 2D")
                    return None
                
                # Add channel dimension [1, height, width]
                return np.expand_dims(matrix, axis=0)
            else:
                print(f"⚠️ Skipping {subject_id}: Missing {self.band_type} file")
                return None
        
        return None
    
    def _get_severity_label(self, subject_id, severity_data):
        """Get severity label with intelligent fallback"""
        # Try exact match first
        if subject_id in severity_data:
            return severity_data[subject_id]
        
        # Try variations
        variations = [
            subject_id.replace('subject_', ''),
            subject_id.replace('_', ''),
            subject_id.split('_')[-1],  # Just the number part
        ]
        
        for variation in variations:
            if variation in severity_data:
                return severity_data[variation]
        
        # Smart default based on subject ID (if numeric)
        try:
            subject_num = int(subject_id.split('_')[-1])
            # Distribute subjects across classes for better balance
            if subject_num % 3 == 0:
                return "Normal"
            elif subject_num % 3 == 1:
                return "Moderate" 
            else:
                return "Severe"
        except:
            return "Normal"  # Final fallback
    
    def _normalize_matrices(self, matrices):
        """Enhanced matrix normalization"""
        print("🔄 Normalizing matrices...")
        
        normalized_matrices = []
        for matrix in matrices:
            norm_matrix = np.zeros_like(matrix)
            
            for c in range(matrix.shape[0]):  # For each channel
                channel = matrix[c]
                
                # Use robust normalization (percentile-based)
                p5, p95 = np.percentile(channel, [5, 95])
                if p95 > p5:
                    norm_channel = (channel - p5) / (p95 - p5)
                    norm_channel = np.clip(norm_channel, 0, 1)
                else:
                    norm_channel = channel
                
                norm_matrix[c] = norm_channel
            
            normalized_matrices.append(norm_matrix)
        
        return np.array(normalized_matrices)
    
    def _discretize_severity(self, severity_scores):
        """Enhanced severity discretization"""
        print("🔄 Discretizing severity scores...")
        
        # Enhanced mapping with more variations
        severity_mapping = {
            # Normal cases
            'Normal': 0, 'normal': 0, 'NORMAL': 0,
            'Control': 0, 'control': 0, 'CONTROL': 0,
            'Healthy': 0, 'healthy': 0, 'HEALTHY': 0,
            'HC': 0, 'hc': 0,  # Healthy Control
            '0': 0, 0: 0,
            
            # Mild cases  
            'Mild': 0, 'mild': 0, 'MILD': 0,
            'Light': 0, 'light': 0, 'LIGHT': 0,
            'Low': 0, 'low': 0, 'LOW': 0,
            '1': 0, 1: 0,
            
            # Moderate cases
            'Moderate': 1, 'moderate': 1, 'MODERATE': 1,
            'Medium': 1, 'medium': 1, 'MEDIUM': 1,
            'Mid': 1, 'mid': 1, 'MID': 1,
            '2': 1, 2: 1,
            
            # Severe cases
            'Severe': 2, 'severe': 2, 'SEVERE': 2,
            'Heavy': 2, 'heavy': 2, 'HEAVY': 2,
            'High': 2, 'high': 2, 'HIGH': 2,
            'Strong': 2, 'strong': 2, 'STRONG': 2,
            '3': 2, 3: 2,
        }
        
        discrete_scores = []
        unmapped_labels = set()
        
        for score in severity_scores:
            # Handle different data types
            if isinstance(score, (int, float)):
                if np.isnan(score):
                    discrete_scores.append(0)  # Default to normal
                    continue
                score_str = str(int(score)) if score == int(score) else str(score)
            else:
                score_str = str(score).strip()
            
            if score_str in severity_mapping:
                discrete_scores.append(severity_mapping[score_str])
            else:
                # Intelligent categorization for unknown labels
                score_lower = score_str.lower()
                
                if any(word in score_lower for word in ['normal', 'control', 'healthy', 'hc']):
                    discrete_scores.append(0)
                elif any(word in score_lower for word in ['mild', 'light', 'low']):
                    discrete_scores.append(0)  # Group mild with normal
                elif any(word in score_lower for word in ['severe', 'heavy', 'high', 'strong']):
                    discrete_scores.append(2)
                elif any(word in score_lower for word in ['moderate', 'medium', 'mid']):
                    discrete_scores.append(1)
                else:
                    # Try numeric interpretation
                    try:
                        numeric_score = float(score_str)
                        if numeric_score <= 1:
                            discrete_scores.append(0)
                        elif numeric_score <= 2:
                            discrete_scores.append(1)
                        else:
                            discrete_scores.append(2)
                    except:
                        discrete_scores.append(0)  # Default to normal
                        unmapped_labels.add(score_str)
        
        if unmapped_labels:
            print(f"⚠️ Unmapped labels (defaulted to Normal): {unmapped_labels}")
        
        discrete_scores = np.array(discrete_scores, dtype=int)
        
        # Print distribution
        unique_discrete = np.unique(discrete_scores)
        print(f"📊 Final severity distribution:")
        severity_names = ['Normal/Mild', 'Moderate', 'Severe']
        for i in range(3):
            count = np.sum(discrete_scores == i)
            print(f"  {severity_names[i]}: {count} samples")
        
        return discrete_scores
    
    def _balance_classes(self):
        """Balance classes using intelligent oversampling"""
        print("⚖️ Balancing classes...")
        
        # Count samples per class
        unique_labels, counts = np.unique(self.severity_labels, return_counts=True)
        max_samples = max(max(counts), self.min_samples_per_class)
        
        balanced_matrices = []
        balanced_labels = []
        balanced_ids = []
        
        for class_idx in range(3):  # 0, 1, 2
            class_mask = self.severity_labels == class_idx
            class_matrices = self.matrices[class_mask]
            class_labels = self.severity_labels[class_mask]
            class_ids = [self.subject_ids[i] for i in range(len(self.subject_ids)) if class_mask[i]]
            
            current_count = len(class_matrices)
            needed_count = max_samples - current_count
            
            if needed_count > 0:
                print(f"  Class {class_idx}: {current_count} -> {max_samples} samples")
                
                # Generate additional samples through augmentation
                additional_matrices = []
                additional_labels = []
                additional_ids = []
                
                for _ in range(needed_count):
                    # Select random sample from existing ones
                    idx = np.random.randint(0, current_count)
                    base_matrix = class_matrices[idx].copy()
                    
                    # Apply severity-specific augmentation
                    augmented_matrix = AdvancedConnectivityAugmentation.severity_specific_augmentation(
                        base_matrix, class_idx
                    )
                    
                    additional_matrices.append(augmented_matrix)
                    additional_labels.append(class_idx)
                    additional_ids.append(f"{class_ids[idx]}_aug_{len(additional_matrices)}")
                
                # Combine original and augmented
                class_matrices = np.concatenate([class_matrices, np.array(additional_matrices)], axis=0)
                class_labels = np.concatenate([class_labels, np.array(additional_labels)], axis=0)
                class_ids.extend(additional_ids)
            
            balanced_matrices.append(class_matrices)
            balanced_labels.append(class_labels)
            balanced_ids.extend(class_ids)
        
        # Combine all classes
        final_matrices = np.concatenate(balanced_matrices, axis=0)
        final_labels = np.concatenate(balanced_labels, axis=0)
        
        print(f"✅ Balanced dataset: {len(final_matrices)} total samples")
        return final_matrices, final_labels, balanced_ids
    
    def _augment_data(self):
        """Apply data augmentation to increase dataset size"""
        if self.augment_factor <= 1:
            return self.matrices, self.severity_labels, self.subject_ids
        
        print(f"🔄 Augmenting data with factor {self.augment_factor}...")
        
        augmented_matrices = [self.matrices]  # Start with original
        augmented_labels = [self.severity_labels]
        augmented_ids = [self.subject_ids]
        
        for aug_round in range(1, math.ceil(self.augment_factor)):
            round_matrices = []
            round_labels = []
            round_ids = []
            
            for i, (matrix, label) in enumerate(zip(self.matrices, self.severity_labels)):
                # Apply random augmentation
                augmentation_methods = [
                    AdvancedConnectivityAugmentation.gaussian_noise,
                    AdvancedConnectivityAugmentation.connectivity_dropout,
                    AdvancedConnectivityAugmentation.smooth_augmentation,
                    AdvancedConnectivityAugmentation.regional_scaling,
                    AdvancedConnectivityAugmentation.severity_specific_augmentation,
                ]
                
                # Choose random augmentation method
                aug_method = np.random.choice(augmentation_methods)
                
                if aug_method == AdvancedConnectivityAugmentation.severity_specific_augmentation:
                    augmented_matrix = aug_method(matrix, label)
                else:
                    augmented_matrix = aug_method(matrix)
                
                round_matrices.append(augmented_matrix)
                round_labels.append(label)
                round_ids.append(f"{self.subject_ids[i]}_aug_r{aug_round}")
            
            augmented_matrices.append(np.array(round_matrices))
            augmented_labels.append(np.array(round_labels))
            augmented_ids.append(round_ids)
        
        # Combine all augmented data
        final_matrices = np.concatenate(augmented_matrices, axis=0)
        final_labels = np.concatenate(augmented_labels, axis=0)
        final_ids = []
        for id_list in augmented_ids:
            final_ids.extend(id_list)
        
        print(f"✅ Augmented dataset: {len(self.matrices)} -> {len(final_matrices)} samples")
        return final_matrices, final_labels, final_ids
    
    def _print_dataset_info(self):
        """Enhanced dataset information printing"""
        print("\n" + "="*60)
        print("📊 ENHANCED EEG CONNECTIVITY DATASET INFO")
        print("="*60)
        print(f"📁 Data directory: {self.data_dir}")
        print(f"📊 Band type: {self.band_type}")
        print(f"👥 Number of subjects: {len(self.matrices)}")
        
        if len(self.matrices) > 0:
            print(f"🧠 Matrix shape: {self.matrices[0].shape}")
            print(f"📏 Matrix value range: [{np.min(self.matrices):.3f}, {np.max(self.matrices):.3f}]")
            
            # Detailed severity distribution
            unique_labels, counts = np.unique(self.severity_labels, return_counts=True)
            severity_names = ['Normal/Mild', 'Moderate', 'Severe']
            
            print(f"\n📈 SEVERITY DISTRIBUTION:")
            total_samples = len(self.severity_labels)
            for i, (label, count) in enumerate(zip(unique_labels, counts)):
                percentage = (count / total_samples) * 100
                print(f"  {severity_names[int(label)]}: {count} samples ({percentage:.1f}%)")
            
            # Connectivity statistics by severity
            print(f"\n📊 CONNECTIVITY STATISTICS:")
            for i in range(3):
                mask = self.severity_labels == i
                if np.any(mask):
                    matrices_subset = self.matrices[mask]
                    avg_connectivity = np.mean(matrices_subset)
                    std_connectivity = np.std(matrices_subset)
                    print(f"  {severity_names[i]}: μ={avg_connectivity:.4f}, σ={std_connectivity:.4f}")
            
            # Data quality metrics
            print(f"\n🔍 DATA QUALITY:")
            print(f"  Augmentation applied: {self.augment}")
            print(f"  Classes balanced: {self.balance_classes}")
            print(f"  Normalization: {self.normalize}")
            
            # Check for potential issues
            if np.any(np.isnan(self.matrices)):
                print(f"  ⚠️ Contains NaN values")
            if np.any(np.isinf(self.matrices)):
                print(f"  ⚠️ Contains infinite values")
            
            # Matrix symmetry check
            symmetry_errors = []
            for i, matrix in enumerate(self.matrices[:min(10, len(self.matrices))]):  # Check first 10
                for c in range(matrix.shape[0]):
                    if matrix.shape[1] == matrix.shape[2]:  # Square matrix
                        symmetry_error = np.mean(np.abs(matrix[c] - matrix[c].T))
                        symmetry_errors.append(symmetry_error)
            
            if symmetry_errors:
                avg_symmetry_error = np.mean(symmetry_errors)
                print(f"  Avg symmetry error: {avg_symmetry_error:.6f}")
        
        print("="*60)
    
    def __len__(self):
        return len(self.matrices)
    
    def __getitem__(self, idx):
        matrix = torch.FloatTensor(self.matrices[idx])
        severity = torch.tensor(self.severity_labels[idx],dtype=torch.long)
        
        # Real-time augmentation during training
        if self.augment and hasattr(self, '_training') and self._training:
            if np.random.random() > 0.7:  # 30% chance
                matrix_np = matrix.numpy()
                matrix_np = AdvancedConnectivityAugmentation.severity_specific_augmentation(
                    matrix_np, self.severity_labels[idx]
                )
                matrix = torch.FloatTensor(matrix_np)
        
        return {
            'adjacency_matrix': matrix,
            'severity_label': severity,
            'subject_id': self.subject_ids[idx]
        }
    
    def train(self):
        """Set dataset to training mode"""
        self._training = True
        return self
    
    def eval(self):
        """Set dataset to evaluation mode"""
        self._training = False
        return self
    
    def get_class_weights(self):
        """Calculate class weights for loss function"""
        unique_labels, counts = np.unique(self.severity_labels, return_counts=True)
        total_samples = len(self.severity_labels)
        
        class_weights = {}
        for label, count in zip(unique_labels, counts):
            class_weights[int(label)] = total_samples / (len(unique_labels) * count)
        
        # Ensure all classes are represented
        weights_tensor = torch.tensor([class_weights.get(i, 1.0) for i in range(3)])
        return weights_tensor
    
    def visualize_augmentation_effects(self, subject_idx=0, save_path=None):
        """Visualize the effects of different augmentation techniques"""
        if subject_idx >= len(self.matrices):
            print("Invalid subject index")
            return
        
        original_matrix = self.matrices[subject_idx]
        severity_label = self.severity_labels[subject_idx]
        
        # Apply different augmentations
        augmentations = {
            'Original': original_matrix,
            'Gaussian Noise': AdvancedConnectivityAugmentation.gaussian_noise(original_matrix.copy()),
            'Connectivity Dropout': AdvancedConnectivityAugmentation.connectivity_dropout(original_matrix.copy()),
            'Smooth Augmentation': AdvancedConnectivityAugmentation.smooth_augmentation(original_matrix.copy()),
            'Regional Scaling': AdvancedConnectivityAugmentation.regional_scaling(original_matrix.copy()),
            'Severity Specific': AdvancedConnectivityAugmentation.severity_specific_augmentation(original_matrix.copy(), severity_label),
        }
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        severity_names = ['Normal/Mild', 'Moderate', 'Severe']
        
        for i, (aug_name, aug_matrix) in enumerate(augmentations.items()):
            if i >= 6:
                break
                
            # Average across channels for visualization
            if aug_matrix.shape[0] > 1:
                display_matrix = np.mean(aug_matrix, axis=0)
            else:
                display_matrix = aug_matrix[0]
            
            im = axes[i].imshow(display_matrix, cmap='viridis', vmin=0, vmax=1, aspect='auto')
            axes[i].set_title(f'{aug_name}\n{severity_names[severity_label]} - Mean: {np.mean(display_matrix):.3f}')
            axes[i].set_xlabel('Brain Region')
            axes[i].set_ylabel('Brain Region')
            plt.colorbar(im, ax=axes[i], shrink=0.8)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Augmentation visualization saved: {save_path}")
        plt.show()

def load_enhanced_eeg_data(data_dir, excel_file_path=None, band_type='both', 
                          batch_size=8, test_split=0.2, augment_factor=2, 
                          balance_classes=True, min_samples_per_class=10):
    """
    Enhanced convenience function to load EEG data with advanced preprocessing
    """
    print(f"🔄 Loading Enhanced EEG data from {data_dir}")
    
    # Create enhanced dataset
    dataset = EnhancedEEGConnectivityDataset(
        data_dir=data_dir,
        excel_file_path=excel_file_path,
        band_type=band_type,
        normalize=True,
        augment=True,
        augment_factor=augment_factor,
        balance_classes=balance_classes,
        min_samples_per_class=min_samples_per_class
    )
    
    if len(dataset) == 0:
        raise ValueError("No data found! Please check your data directory and file patterns.")
    
    # Smart train/test splitting
    indices = list(range(len(dataset)))
    unique_labels, counts = np.unique(dataset.severity_labels, return_counts=True)
    min_count = np.min(counts)
    
    if min_count >= 2:
        # Use stratified split if possible
        try:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=42)
            train_indices, test_indices = next(splitter.split(indices, dataset.severity_labels))
            print(f"✅ Using stratified split")
        except:
            # Fallback to random split
            train_indices, test_indices = train_test_split(
                indices, test_size=test_split, random_state=42
            )
            print(f"⚠️ Using random split (stratification failed)")
    else:
        # Random split for very small datasets
        train_indices, test_indices = train_test_split(
            indices, test_size=test_split, random_state=42
        )
        print(f"⚠️ Using random split due to insufficient samples per class")
    
    # Create train and test datasets with consistent settings
    train_dataset = EnhancedEEGConnectivityDataset(
        data_dir=data_dir,
        excel_file_path=excel_file_path,
        band_type=band_type,
        normalize=True,
        augment=True,  # Augmentation for training
        augment_factor=augment_factor,  # Use the same augment factor
        balance_classes=balance_classes,
        min_samples_per_class=min_samples_per_class
    )
    
    test_dataset = EnhancedEEGConnectivityDataset(
        data_dir=data_dir,
        excel_file_path=excel_file_path,
        band_type=band_type,
        normalize=True,
        augment=False,  # No augmentation for testing
        augment_factor=1,
        balance_classes=False,  # Keep original distribution for testing
        min_samples_per_class=0
    )
    
    # Create subset datasets using the original dataset's indices
    from torch.utils.data import Subset
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)
    
    # Create dataloaders with enhanced settings
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,  # Set to 0 to avoid worker issues during debugging
        pin_memory=True,
        persistent_workers=False
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,  # Set to 0 to avoid worker issues during debugging
        pin_memory=True,
        persistent_workers=False
    )
    
    print(f"✅ Enhanced data loaded successfully!")
    print(f"📊 Train samples: {len(train_subset)}, Test samples: {len(test_subset)}")
    print(f"🎯 Class weights available: {dataset.get_class_weights()}")
    
    return train_loader, test_loader, dataset

# Enhanced data analysis functions
class DatasetAnalyzer:
    """Advanced dataset analysis tools"""
    
    @staticmethod
    def analyze_class_distribution(dataset):
        """Analyze class distribution and suggest improvements"""
        labels = dataset.severity_labels
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        print("\n📊 CLASS DISTRIBUTION ANALYSIS")
        print("-" * 40)
        
        total = len(labels)
        severity_names = ['Normal/Mild', 'Moderate', 'Severe']
        
        for label, count in zip(unique_labels, counts):
            percentage = (count / total) * 100
            print(f"{severity_names[int(label)]}: {count:3d} ({percentage:5.1f}%)")
        
        # Calculate imbalance ratio
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"\nImbalance ratio: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 3:
            print("⚠️  HIGH IMBALANCE - Consider:")
            print("   • Increasing augmentation factor")
            print("   • Using balanced sampling")
            print("   • Collecting more minority class data")
        elif imbalance_ratio > 1.5:
            print("⚠️  MODERATE IMBALANCE - Consider:")
            print("   • Enabling class balancing")
            print("   • Using class weights in loss function")
        else:
            print("✅ WELL BALANCED")
    
    @staticmethod
    def analyze_matrix_properties(dataset, num_samples=10):
        """Analyze connectivity matrix properties"""
        print("\n🔍 MATRIX PROPERTIES ANALYSIS")
        print("-" * 40)
        
        matrices = dataset.matrices[:num_samples]
        
        # Basic statistics
        print(f"Matrix shape: {matrices[0].shape}")
        print(f"Value range: [{np.min(matrices):.4f}, {np.max(matrices):.4f}]")
        print(f"Mean connectivity: {np.mean(matrices):.4f}")
        print(f"Std connectivity: {np.std(matrices):.4f}")
        
        # Symmetry analysis
        symmetry_errors = []
        for matrix in matrices:
            for c in range(matrix.shape[0]):
                if matrix.shape[1] == matrix.shape[2]:
                    error = np.mean(np.abs(matrix[c] - matrix[c].T))
                    symmetry_errors.append(error)
        
        print(f"Avg symmetry error: {np.mean(symmetry_errors):.6f}")
        
        # Sparsity analysis
        sparsity_levels = []
        for matrix in matrices:
            for c in range(matrix.shape[0]):
                sparsity = np.sum(matrix[c] < 0.1) / matrix[c].size
                sparsity_levels.append(sparsity)
        
        print(f"Avg sparsity (< 0.1): {np.mean(sparsity_levels):.2f}")
        
    @staticmethod
    def suggest_improvements(dataset):
        """Suggest improvements based on dataset analysis"""
        print("\n💡 IMPROVEMENT SUGGESTIONS")
        print("-" * 40)
        
        # Analyze class distribution
        labels = dataset.severity_labels
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        total_samples = len(labels)
        min_count = min(counts) if len(counts) > 0 else 0
        
        if total_samples < 50:
            print("📈 INCREASE DATA SIZE:")
            print("   • Set augment_factor=3 or higher")
            print("   • Enable balance_classes=True")
            print("   • Consider synthetic data generation")
        
        if min_count < 5:
            print("⚖️ ADDRESS CLASS IMBALANCE:")
            print("   • Set min_samples_per_class=10")
            print("   • Use weighted loss functions")
            print("   • Collect more minority class data")
        
        # Check matrix quality
        matrices = dataset.matrices
        if len(matrices) > 0:
            if np.any(np.isnan(matrices)) or np.any(np.isinf(matrices)):
                print("🔧 FIX DATA QUALITY:")
                print("   • Check for NaN/inf values in source data")
                print("   • Improve data preprocessing")
        
        print("✅ GENERAL RECOMMENDATIONS:")
        print("   • Use enhanced data loader")
        print("   • Enable real-time augmentation")
        print("   • Monitor training/validation curves")

# Example usage and testing
if __name__ == "__main__":
    # Example paths - update these for your data
    DATA_DIR = r"E:\Desktop\IEEE\PD_all_adjacency_files_updated\PD_all_adjacency_files_updated"
    EXCEL_FILE = r"E:\Desktop\IEEE\PD_TBR_Classification_Updated.xlsx"
    
    try:
        print("🚀 Testing Enhanced EEG Data Loader...")
        
        # Load data with enhanced features
        train_loader, test_loader, full_dataset = load_enhanced_eeg_data(
            data_dir=DATA_DIR,
            excel_file_path=EXCEL_FILE,
            band_type='both',
            batch_size=4,
            test_split=0.2,
            augment_factor=2,
            balance_classes=True,
            min_samples_per_class=8
        )
        
        # Analyze dataset
        DatasetAnalyzer.analyze_class_distribution(full_dataset)
        DatasetAnalyzer.analyze_matrix_properties(full_dataset)
        DatasetAnalyzer.suggest_improvements(full_dataset)
        
        # Test data loading
        print("\n🧪 Testing data loading...")
        for batch_idx, batch in enumerate(train_loader):
            matrices = batch['adjacency_matrix']
            labels = batch['severity_label']
            
            print(f"Batch {batch_idx}: Matrix {matrices.shape}, Labels {labels.flatten().tolist()}")
            
            if batch_idx >= 2:
                break
        
        # Visualize augmentation effects
        full_dataset.visualize_augmentation_effects(0)
        
        print("\n✅ Enhanced EEG Data Loader test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n📝 Please update the DATA_DIR and EXCEL_FILE paths to match your data location.")
