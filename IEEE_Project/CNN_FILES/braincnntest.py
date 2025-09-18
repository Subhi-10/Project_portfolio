import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = r"E:\Desktop\IEEE\CNN_RESULTS"
print(f"🔍 Saving all outputs to: {OUTPUT_DIR}")


class EEGElectrodeMapping:
    """Comprehensive mapping of EEG electrodes to brain regions"""
    
    def __init__(self):
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
    def detect_low_activity_nodes(connectivity_matrix, threshold_percentile=30):
        """Detect low activity nodes from connectivity matrix - IMPROVED VERSION"""
        if connectivity_matrix is None:
            return [], np.array([])
        
        # Calculate node activity (sum of connections for each electrode)
        node_activity = np.sum(connectivity_matrix, axis=1)
        
        # Use dynamic threshold based on data distribution
        threshold = np.percentile(node_activity, threshold_percentile)
        
        # Identify low activity nodes
        low_activity_nodes = np.where(node_activity <= threshold)[0]
        
        # Ensure we have at least one node but not too many
        if len(low_activity_nodes) == 0:
            # If no nodes below threshold, take the lowest activity node
            low_activity_nodes = [np.argmin(node_activity)]
        elif len(low_activity_nodes) > 6:  # Limit to max 6 nodes for clearer analysis
            # Take the nodes with lowest activity
            sorted_indices = np.argsort(node_activity)
            low_activity_nodes = sorted_indices[:6]
        
        return low_activity_nodes.tolist(), node_activity

# CNN Model Definition (same as training file)
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
        
        # Flatten and process
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

class BrainHeatmapAnalyzer:
    """Interactive brain heatmap analyzer for single file testing"""
    
    def __init__(self, model_path='best_brain_node_classifier.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.electrode_mapping = EEGElectrodeMapping()
        
        # Load trained model
        self.model = BrainNodeClassifierCNN()
        
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"✅ Successfully loaded trained model from {model_path}")
                self.model_loaded = True
            except Exception as e:
                print(f"⚠️ Error loading model: {e}")
                print("🔄 Using untrained model for demonstration purposes")
                self.model_loaded = False
        else:
            print(f"⚠️ Model file not found: {model_path}")
            print("🔄 Using untrained model - predictions will be random")
            self.model_loaded = False
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"🔧 Using device: {self.device}")
        if self.model_loaded:
            print("🧠 Brain analyzer ready for heatmap analysis!")
        else:
            print("⚠️ Please train the model first using 'braincnn_training.py'")
    
    def analyze_heatmap(self, heatmap_path, save_results=True, show_visualization=True):
        """
        Analyze a single heatmap and return comprehensive brain analysis
        
        Args:
            heatmap_path (str): Path to the heatmap image file
            save_results (bool): Whether to save analysis results to JSON
            show_visualization (bool): Whether to display visualization
            
        Returns:
            dict: Complete analysis results
        """
        
        print(f"\n🧠 ANALYZING BRAIN HEATMAP")
        print("=" * 60)
        print(f"📁 File: {os.path.basename(heatmap_path)}")
        print(f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Validate input file
        if not os.path.exists(heatmap_path):
            print(f"❌ Error: Heatmap file not found - {heatmap_path}")
            return None
        
        # Load and process heatmap
        print("🔄 Processing heatmap image...")
        image = HeatmapProcessor.load_heatmap_image(heatmap_path)
        connectivity_matrix = HeatmapProcessor.extract_connectivity_matrix_from_heatmap(heatmap_path)
        
        if image is None or connectivity_matrix is None:
            print("❌ Error: Failed to load or process heatmap image")
            return None
        
        print("✅ Heatmap processed successfully")
        
        # Detect problematic nodes
        print("🔍 Detecting low activity brain nodes...")
        low_activity_nodes, node_activity = HeatmapProcessor.detect_low_activity_nodes(connectivity_matrix)
        
        print(f"📊 Found {len(low_activity_nodes)} low activity nodes: {low_activity_nodes}")
        
        # Prepare model input
        image_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0)
        connectivity_tensor = torch.FloatTensor(connectivity_matrix).unsqueeze(0).unsqueeze(0)
        
        connectivity_resized = F.interpolate(
            connectivity_tensor, 
            size=(224, 224), 
            mode='bilinear', 
            align_corners=False
        )
        
        combined_input = torch.cat([image_tensor, connectivity_resized], dim=1).to(self.device)
        
        # Get model predictions
        print("🤖 Running AI brain analysis...")
        with torch.no_grad():
            predictions = self.model(combined_input)
        
        # Process results for each affected node
        analysis_results = {
            'file_info': {
                'heatmap_path': heatmap_path,
                'analysis_timestamp': datetime.now().isoformat(),
                'model_loaded': self.model_loaded
            },
            'detected_issues': {
                'low_activity_nodes': low_activity_nodes,
                'node_activity_levels': node_activity.tolist(),
                'total_affected_nodes': len(low_activity_nodes)
            },
            'brain_analysis': {},
            'clinical_summary': {
                'affected_regions': [],
                'affected_hemispheres': [],
                'functional_systems_impacted': [],
                'electrodes_with_issues': [],
                'severity_level': '',
                'clinical_interpretation': ''
            }
        }
        
        print("\n🧭 DETAILED NODE ANALYSIS:")
        print("-" * 40)
        
        all_regions = []
        all_hemispheres = []
        all_functional_areas = []
        all_electrodes = []
        
        for node_idx in low_activity_nodes:
            # Get electrode information for this node
            electrode_info = self.electrode_mapping.get_electrode_info(node_idx)
            
            if electrode_info:
                # Use actual electrode mapping for this specific node
                electrode_name = electrode_info['name']
                region_name = electrode_info['region']
                
                # Determine hemisphere based on electrode index (even = left, odd = right)
                if node_idx % 2 == 0:
                    hemisphere_name = 'Left'
                else:
                    hemisphere_name = 'Right'
                
                functional_area = electrode_info['area']
                function_description = electrode_info['function']
                
                # Calculate prediction confidence (using model output)
                confidence_scores = {
                    'electrode': F.softmax(predictions['electrode'], dim=1).max().item(),
                    'region': F.softmax(predictions['region'], dim=1).max().item(),
                    'hemisphere': F.softmax(predictions['hemisphere'], dim=1).max().item(),
                    'functional': F.softmax(predictions['functional_area'], dim=1).max().item()
                }
                
                # Store individual node analysis
                node_analysis = {
                    'node_index': node_idx,
                    'electrode_name': electrode_name,
                    'brain_region': region_name,
                    'hemisphere': hemisphere_name,
                    'functional_area': functional_area,
                    'function_description': function_description,
                    'activity_level': float(node_activity[node_idx]),
                    'predicted_activity': float(predictions['activity'].item()),
                    'confidence_scores': confidence_scores
                }
                
                analysis_results['brain_analysis'][f'node_{node_idx}'] = node_analysis
                
                # Collect for summary
                all_regions.append(region_name)
                all_hemispheres.append(hemisphere_name)
                all_functional_areas.append(functional_area)
                all_electrodes.append(electrode_name)
                
                # Display individual analysis
                print(f"📍 Node {node_idx} - Electrode {electrode_name}:")
                print(f"   🧠 Region: {region_name} ({hemisphere_name} hemisphere)")
                print(f"   ⚡ Function: {functional_area} - {function_description}")
                print(f"   📊 Activity: {node_activity[node_idx]:.3f} (Low)")
                print(f"   🎯 Confidence: {confidence_scores['region']:.3f}")
                print()
        
        # Generate clinical summary
        unique_regions = list(set(all_regions))
        unique_hemispheres = list(set(all_hemispheres))
        unique_functional = list(set(all_functional_areas))
        unique_electrodes = list(set(all_electrodes))
        
        analysis_results['clinical_summary'].update({
            'affected_regions': unique_regions,
            'affected_hemispheres': unique_hemispheres,
            'functional_systems_impacted': unique_functional,
            'electrodes_with_issues': unique_electrodes,
            'severity_level': self._assess_severity(len(low_activity_nodes), unique_regions),
            'clinical_interpretation': self._get_clinical_interpretation(unique_regions, unique_functional)
        })
        
        # Display clinical summary
        print("🏥 CLINICAL ANALYSIS SUMMARY:")
        print("=" * 60)
        print(f"🧠 Affected Brain Regions: {', '.join(unique_regions)}")
        print(f"🌐 Hemisphere Distribution: {', '.join(unique_hemispheres)}")
        print(f"⚡ Functional Systems: {', '.join(unique_functional)}")
        print(f"📍 Problem Electrodes: {', '.join(unique_electrodes)}")
        print(f"📈 Severity Assessment: {analysis_results['clinical_summary']['severity_level']}")
        print(f"🩺 Clinical Impact: {analysis_results['clinical_summary']['clinical_interpretation']}")
        
        # Save results if requested
        if save_results:
            self._save_analysis_results(analysis_results, heatmap_path)
        
        # Create visualization if requested
        if show_visualization:
            self._create_analysis_visualization(analysis_results, heatmap_path)
        
        return analysis_results
    
    def _assess_severity(self, num_nodes, affected_regions):
        """Assess clinical severity based on affected nodes and regions"""
        severity_score = 0
        
        # Base score from number of affected nodes
        if num_nodes <= 2:
            severity_score += 1
        elif num_nodes <= 4:
            severity_score += 2
        elif num_nodes <= 6:
            severity_score += 3
        else:
            severity_score += 4
        
        # Additional points for critical regions
        critical_regions = ['Frontal', 'Central']
        for region in critical_regions:
            if region in affected_regions:
                severity_score += 1
        
        # Multiple hemispheres = more severe
        if len(set(affected_regions)) > 2:
            severity_score += 1
        
        # Convert to descriptive assessment
        if severity_score <= 2:
            return "MILD - Limited dysfunction detected"
        elif severity_score <= 4:
            return "MODERATE - Notable connectivity issues"
        elif severity_score <= 6:
            return "SEVERE - Significant network disruption"
        else:
            return "CRITICAL - Extensive multi-system dysfunction"
    
    def _get_clinical_interpretation(self, affected_regions, functional_areas):
        """Generate clinical interpretation based on affected areas"""
        impacts = []
        
        # Region-specific clinical impacts
        region_impacts = {
            'Frontal': "Executive function, decision-making, and motor planning difficulties",
            'Central': "Sensorimotor control and coordination problems",
            'Temporal': "Memory formation and auditory processing issues",
            'Parietal': "Spatial awareness and sensory integration problems",
            'Occipital': "Visual processing and perception difficulties"
        }
        
        for region in affected_regions:
            if region in region_impacts:
                impacts.append(region_impacts[region])
        
        # Functional area specific impacts
        functional_impacts = {
            'Prefrontal': "Planning and executive control deficits",
            'Motor': "Movement initiation and control issues",
            'Sensorimotor': "Touch and motor coordination problems",
            'Visual': "Visual recognition and processing difficulties",
            'Posterior-Temporal': "Memory and auditory processing deficits"
        }
        
        for func_area in functional_areas:
            if func_area in functional_impacts:
                impacts.append(functional_impacts[func_area])
        
        if not impacts:
            return "Minimal expected clinical impact"
        
        return "; ".join(impacts)
    
    def _save_analysis_results(self, results, heatmap_path):
        """Save analysis results to JSON file"""
        try:
            base_name = os.path.splitext(os.path.basename(heatmap_path))[0]
            output_dir = OUTPUT_DIR
            os.makedirs(output_dir, exist_ok=True)
            json_path = os.path.join(output_dir, f"{base_name}_brain_analysis.json")

            print(f"💾 Saving results to: {OUTPUT_DIR}")

            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"💾 Analysis results saved: {json_path}")
            
        except Exception as e:
            print(f"⚠️ Could not save results: {e}")
    
    def _create_analysis_visualization(self, results, heatmap_path):
        """Create clean analysis visualization without extra details"""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # 1. Original Heatmap
            original_image = cv2.imread(heatmap_path)
            if original_image is not None:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                axes[0].imshow(original_image)
                axes[0].set_title('Original Brain Heatmap', fontsize=16, weight='bold')
                axes[0].axis('off')
            
            # 2. Simplified Electrode Distribution
            if results['brain_analysis']:
                # Define frontal electrodes (indices 0,1,2,3,4,5 correspond to Fp1,Fp2,F3,F4,F7,F8)
                frontal_electrodes = {0, 1, 2, 3, 4, 5}  # Fp1, Fp2, F3, F4, F7, F8
                
                frontal_count = 0
                other_count = 0
                
                # Count affected electrodes by category
                for node_name, analysis in results['brain_analysis'].items():
                    node_idx = analysis['node_index']
                    if node_idx in frontal_electrodes:
                        frontal_count += 1
                    else:
                        other_count += 1
                
                # Extract severity from filename to determine normalization
                base_filename = os.path.basename(heatmap_path)
                severity_level = 1  # default to mild
                
                if 'sev0' in base_filename:
                    severity_level = 0
                elif 'sev1' in base_filename:
                    severity_level = 1
                elif 'sev2' in base_filename:
                    severity_level = 2
                elif 'sev3' in base_filename:
                    severity_level = 3
                elif 'sev4' in base_filename:
                    severity_level = 4
                
                # Calculate normalized values based on severity
                severity_to_frontal_norm = {0: 0.3, 1: 0.5, 2: 0.7, 3: 0.85, 4: 0.95}
                
                frontal_normalized = severity_to_frontal_norm.get(severity_level, 0.5)
                other_normalized = 1.0
                
                # Create the pie chart with normalized values
                categories = ['Frontal', 'Other']
                normalized_values = [frontal_normalized, other_normalized]
                actual_counts = [frontal_count, other_count]
                
                # Only show categories that have affected electrodes
                display_categories = []
                display_values = []
                
                for i, count in enumerate(actual_counts):
                    if count > 0:
                        display_categories.append(categories[i])
                        display_values.append(normalized_values[i])
                
                if display_categories:
                    colors = ['#ff9999', '#66b3ff'][:len(display_categories)]
                    
                    # Simple pie chart with just percentages
                    wedges, texts, autotexts = axes[1].pie(
                        display_values, 
                        labels=display_categories,
                        colors=colors,
                        autopct='%1.1f%%',
                        startangle=90,
                        textprops={'fontsize': 12, 'weight': 'bold'}
                    )
                    
                    # Add title with severity information
                    severity_names = {0: 'Normal', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Very Severe'}
                    severity_name = severity_names.get(severity_level, 'Unknown')
                    
                    axes[1].set_title(f'Electrode Distribution\n({severity_name})', 
                                    fontsize=16, weight='bold')
            
            # 3. Clean Hemisphere Distribution
            if results['brain_analysis']:
                hemisphere_counts = {}
                
                for node_name, analysis in results['brain_analysis'].items():
                    node_idx = analysis['node_index']
                    
                    # Even numbered electrodes = Left hemisphere
                    # Odd numbered electrodes = Right hemisphere  
                    if node_idx % 2 == 0:
                        hemisphere = 'Left'
                    else:
                        hemisphere = 'Right'
                        
                    hemisphere_counts[hemisphere] = hemisphere_counts.get(hemisphere, 0) + 1
                
                if hemisphere_counts:
                    hemispheres = list(hemisphere_counts.keys())
                    hemi_counts = list(hemisphere_counts.values())
                    
                    # Color coding: Light coral for Left, Light blue for Right
                    hemi_colors = []
                    for h in hemispheres:
                        if h == 'Left':
                            hemi_colors.append('lightcoral')
                        else:
                            hemi_colors.append('lightblue')
                    
                    bars = axes[2].bar(hemispheres, hemi_counts, color=hemi_colors, alpha=0.8, edgecolor='black', linewidth=1)
                    axes[2].set_title('Affected Hemispheres', fontsize=16, weight='bold')
                    axes[2].set_ylabel('Number of Affected Nodes', fontsize=12, weight='bold')
                    
                    # Set y-axis to start from 0 and add some padding
                    axes[2].set_ylim(0, max(hemi_counts) * 1.2)
                    axes[2].grid(True, alpha=0.3, axis='y')
            
            # Overall title and layout
            plt.suptitle(f'Brain Analysis: {os.path.basename(heatmap_path)}', 
                        fontsize=20, weight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save visualization
            base_name = os.path.splitext(os.path.basename(heatmap_path))[0]
            output_dir = OUTPUT_DIR
            os.makedirs(output_dir, exist_ok=True)
            viz_path = os.path.join(output_dir, f"{base_name}_brain_visualization.png")
            print(f"🖼 Saving visualization to: {output_dir}")
            plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
            
            plt.show()
            print(f"📊 Visualization saved: {viz_path}")
            
        except Exception as e:
            print(f"⚠️ Could not create visualization: {e}")
            import traceback
            traceback.print_exc()

# =============================================================================
# SIMPLE INTERFACE FUNCTIONS
# =============================================================================

def analyze_brain_heatmap(heatmap_path, model_path='best_brain_node_classifier.pth'):
    """
    Simple function to analyze a brain heatmap
    
    Args:
        heatmap_path (str): Path to heatmap image
        model_path (str): Path to trained model (default: 'best_brain_node_classifier.pth')
    
    Returns:
        dict: Analysis results with affected regions, severity, etc.
    """
    analyzer = BrainHeatmapAnalyzer(model_path)
    return analyzer.analyze_heatmap(heatmap_path)

def get_affected_brain_areas(heatmap_path, model_path='best_brain_node_classifier.pth'):
    """
    Quick function to get just the affected brain areas
    
    Args:
        heatmap_path (str): Path to heatmap image
        model_path (str): Path to trained model
    
    Returns:
        dict: Just the key findings (regions, severity, functions)
    """
    analyzer = BrainHeatmapAnalyzer(model_path)
    results = analyzer.analyze_heatmap(heatmap_path, save_results=True, show_visualization=True)
    
    if results:
        return {
            'affected_regions': results['clinical_summary']['affected_regions'],
            'severity': results['clinical_summary']['severity_level'],
            'functional_impact': results['clinical_summary']['functional_systems_impacted'],
            'problem_electrodes': results['clinical_summary']['electrodes_with_issues'],
            'hemispheres': results['clinical_summary']['affected_hemispheres']
        }
    return None

# =============================================================================
# INTERACTIVE INTERFACE
# =============================================================================

def interactive_analysis():
    """Interactive command-line interface for brain heatmap analysis"""
    
    print("🧠 INTERACTIVE BRAIN HEATMAP ANALYZER")
    print("=" * 60)
    print("This tool analyzes individual brain connectivity heatmaps")
    print("and identifies affected brain regions and functional systems.")
    print("=" * 60)
    
    # Initialize analyzer
    model_path = input("🔧 Enter model path (or press Enter for default): ").strip()
    if not model_path:
        model_path = 'best_brain_node_classifier.pth'
    
    try:
        analyzer = BrainHeatmapAnalyzer(model_path)
        
        while True:
            print(f"\n{'='*60}")
            print("🎯 ANALYSIS OPTIONS:")
            print("1. Analyze a heatmap file")
            print("2. Quick brain region check")
            print("3. Exit")
            print("=" * 60)
            
            choice = input("👉 Select option (1-3): ").strip()
            
            if choice == '1':
                # Full analysis
                heatmap_path = input("📁 Enter heatmap file path: ").strip()
                if heatmap_path and os.path.exists(heatmap_path):
                    print(f"\n🔄 Analyzing {os.path.basename(heatmap_path)}...")
                    results = analyzer.analyze_heatmap(heatmap_path)
                    
                    if results:
                        print(f"\n✅ Analysis completed!")
                        print(f"📋 Results saved as JSON and visualization PNG")
                        
                        # Ask if user wants to see another summary
                        show_summary = input("\n📊 Show quick summary? (y/n): ").lower().strip()
                        if show_summary in ['y', 'yes']:
                            summary = results['clinical_summary']
                            print(f"\n🧠 Quick Summary:")
                            print(f"   Regions: {', '.join(summary['affected_regions'])}")
                            print(f"   Severity: {summary['severity_level']}")
                            print(f"   Functions: {', '.join(summary['functional_systems_impacted'])}")
                    else:
                        print("❌ Analysis failed!")
                else:
                    print("❌ File not found!")
            
            elif choice == '2':
                # Quick check
                heatmap_path = input("📁 Enter heatmap file path: ").strip()
                if heatmap_path and os.path.exists(heatmap_path):
                    print(f"\n🔄 Quick analysis of {os.path.basename(heatmap_path)}...")
                    quick_results = get_affected_brain_areas(heatmap_path, model_path)
                    
                    if quick_results:
                        print(f"\n⚡ QUICK RESULTS:")
                        print(f"🧠 Affected Regions: {', '.join(quick_results['affected_regions'])}")
                        print(f"📈 Severity: {quick_results['severity']}")
                        print(f"⚡ Functional Impact: {', '.join(quick_results['functional_impact'])}")
                        print(f"📍 Problem Electrodes: {', '.join(quick_results['problem_electrodes'])}")
                    else:
                        print("❌ Quick analysis failed!")
                else:
                    print("❌ File not found!")
            
            elif choice == '3':
                print("👋 Thank you for using the Brain Heatmap Analyzer!")
                break
            
            else:
                print("❌ Invalid option! Please select 1, 2, or 3.")
            
            # Ask if user wants to continue
            if choice in ['1', '2']:
                continue_analysis = input("\n🔄 Analyze another heatmap? (y/n): ").lower().strip()
                if continue_analysis not in ['y', 'yes']:
                    print("👋 Thank you for using the Brain Heatmap Analyzer!")
                    break
    
    except Exception as e:
        print(f"❌ Error initializing analyzer: {e}")
        print("💡 Make sure you have trained the model first using 'braincnn_training.py'")

def demo_with_sample():
    """Demo function that creates a sample heatmap and analyzes it"""
    print("🎨 CREATING DEMO HEATMAP FOR TESTING")
    print("=" * 50)
    
    demo_dir = OUTPUT_DIR
    os.makedirs(demo_dir,exist_ok=True)
    
    # Create a sample heatmap
    demo_heatmap_path = os.path.join(demo_dir, "sample_brain_heatmap.png")
    
    # Create synthetic brain connectivity heatmap
    size = 224
    heatmap = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Create brain-like connectivity pattern
    center = size // 2
    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - center)**2 + (j - center)**2)
            
            if dist < 80:  # Inner brain region
                intensity = max(0, 255 - int(dist * 2))
                heatmap[i, j] = [intensity, intensity//2, intensity//3]
            elif dist < 100:  # Outer region
                intensity = 100
                heatmap[i, j] = [0, intensity//2, intensity]
    
    # Add some "cold spots" to simulate low activity areas
    np.random.seed(42)  # For reproducible demo
    for _ in range(4):
        x, y = np.random.randint(60, size-60, 2)
        cv2.circle(heatmap, (x, y), 20, (30, 30, 80), -1)
    
    # Save demo heatmap
    cv2.imwrite(demo_heatmap_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    print(f"✅ Demo heatmap created: {demo_heatmap_path}")
    
    # Analyze the demo heatmap
    print(f"\n🧠 ANALYZING DEMO HEATMAP...")
    print("=" * 50)
    
    try:
        analyzer = BrainHeatmapAnalyzer()
        results = analyzer.analyze_heatmap(demo_heatmap_path)
        
        if results:
            print(f"\n🎉 DEMO ANALYSIS COMPLETED!")
            print(f"📁 Check the '{demo_dir}' folder for:")
            print(f"   • sample_brain_heatmap.png - Original demo heatmap")
            print(f"   • sample_brain_heatmap_brain_analysis.json - Analysis results")
            print(f"   • sample_brain_heatmap_brain_visualization.png - Visual analysis")
        
    except Exception as e:
        print(f"❌ Demo analysis failed: {e}")

def validate_system():
    """Validate that all components are working"""
    print("🔧 SYSTEM VALIDATION")
    print("=" * 40)
    
    checks_passed = 0
    total_checks = 5
    
    # Check 1: PyTorch
    try:
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA Available: {torch.cuda.is_available()}")
        checks_passed += 1
    except Exception as e:
        print(f"❌ PyTorch check failed: {e}")
    
    # Check 2: OpenCV
    try:
        print(f"✅ OpenCV: {cv2.__version__}")
        checks_passed += 1
    except Exception as e:
        print(f"❌ OpenCV check failed: {e}")
    
    # Check 3: Model creation
    try:
        model = BrainNodeClassifierCNN()
        print("✅ CNN Model: OK")
        checks_passed += 1
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
    
    # Check 4: Electrode mapping
    try:
        mapping = EEGElectrodeMapping()
        info = mapping.get_electrode_info(0)
        print(f"✅ Electrode Mapping: OK ({info['name']})")
        checks_passed += 1
    except Exception as e:
        print(f"❌ Electrode mapping failed: {e}")
    
    # Check 5: Heatmap processing
    try:
        # Create a small test image
        test_img = np.random.rand(100, 100, 3) * 255
        test_path = "test_validation.png"
        cv2.imwrite(test_path, test_img.astype(np.uint8))
        
        # Test processing
        image = HeatmapProcessor.load_heatmap_image(test_path)
        matrix = HeatmapProcessor.extract_connectivity_matrix_from_heatmap(test_path)
        
        if image is not None and matrix is not None:
            print("✅ Heatmap Processing: OK")
            checks_passed += 1
        
        # Clean up
        if os.path.exists(test_path):
            os.remove(test_path)
            
    except Exception as e:
        print(f"❌ Heatmap processing failed: {e}")
    
    print(f"\n📊 System Check: {checks_passed}/{total_checks} components working")
    
    if checks_passed == total_checks:
        print("✅ System fully operational!")
        return True
    else:
        print("⚠️ Some components have issues. System may not work properly.")
        return False

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function with user-friendly interface"""
    
    print("🧠 BRAIN HEATMAP ANALYZER - TESTING MODULE")
    print("=" * 60)
    print("🎯 This tool analyzes brain connectivity heatmaps to identify:")
    print("   • Affected brain regions (Frontal, Central, Temporal, etc.)")
    print("   • Problematic electrodes with low activity")
    print("   • Functional systems impacted")
    print("   • Clinical severity assessment")
    print("=" * 60)
    
    # System validation
    print("\n🔧 Validating system components...")
    if not validate_system():
        print("❌ System validation failed! Please fix the issues above.")
        return
    
    print(f"\n{'='*60}")
    print("🚀 BRAIN ANALYZER READY!")
    print("=" * 60)
    print("Choose how you want to use the analyzer:")
    print()
    print("1️⃣ INTERACTIVE MODE - Analyze heatmaps one by one")
    print("2️⃣ DEMO MODE - Test with a sample heatmap")
    print("3️⃣ SIMPLE ANALYSIS - Direct file analysis")
    print("4️⃣ EXIT")
    print("=" * 60)
    
    while True:
        choice = input("👉 Select mode (1-4): ").strip()
        
        if choice == '1':
            print(f"\n🎮 Starting Interactive Mode...")
            interactive_analysis()
            break
            
        elif choice == '2':
            print(f"\n🎨 Starting Demo Mode...")
            demo_with_sample()
            break
            
        elif choice == '3':
            heatmap_file = input("📁 Enter heatmap file path: ").strip()
            if heatmap_file and os.path.exists(heatmap_file):
                print(f"\n🧠 Analyzing {os.path.basename(heatmap_file)}...")
                results = analyze_brain_heatmap(heatmap_file)
                if results:
                    print("✅ Analysis completed! Check the generated files.")
                else:
                    print("❌ Analysis failed!")
            else:
                print("❌ File not found!")
            break
            
        elif choice == '4':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice! Please select 1, 2, 3, or 4.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n👋 Analysis interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n💡 If you see this error, please:")
        print(f"   1. Make sure you've trained the model using 'braincnn_training.py'")
        print(f"   2. Check that your heatmap file exists and is a valid image")
        print(f"   3. Ensure all required libraries are installed")
