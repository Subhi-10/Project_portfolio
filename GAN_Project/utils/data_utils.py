import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArtifactDataset(Dataset):
    def __init__(self, complete_dir, damaged_dir, image_size=256):
        """
        Args:
            complete_dir (str): Directory with complete images.
            damaged_dir (str): Directory with damaged images.
            image_size (int): Size to resize images to.
        """
        self.complete_dir = complete_dir
        self.damaged_dir = damaged_dir
        self.image_size = image_size
        
        # Get complete file list
        self.complete_images = sorted([f for f in os.listdir(complete_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Map each complete image to its corresponding damaged image
        self.image_pairs = []
        missing_pairs = []

        for complete_img in self.complete_images:
            base_name, ext = os.path.splitext(complete_img)  # Extract filename without extension
            
            # Expected damaged image name
            damaged_img = f"damaged_{base_name}{ext}"
            
            # Check if the corresponding damaged image exists
            if os.path.exists(os.path.join(damaged_dir, damaged_img)):
                self.image_pairs.append((complete_img, damaged_img))
            else:
                missing_pairs.append(complete_img)
        
        logger.info(f"✅ Found {len(self.image_pairs)} valid image pairs.")
        
        # Log missing pairs
        if missing_pairs:
            logger.warning(f"⚠️ Missing {len(missing_pairs)} damaged images! Training may be affected.")
            for img in missing_pairs[:10]:  # Show only first 10 missing pairs
                logger.warning(f"  - {img}")

        if len(self.image_pairs) == 0:
            raise ValueError("❌ No valid image pairs found! Please check your file names.")

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def __len__(self):
        return len(self.image_pairs)
        
    def __getitem__(self, idx):
        complete_img_name, damaged_img_name = self.image_pairs[idx]
        
        complete_path = os.path.join(self.complete_dir, complete_img_name)
        damaged_path = os.path.join(self.damaged_dir, damaged_img_name)
        
        try:
            complete_img = Image.open(complete_path).convert('RGB')
            damaged_img = Image.open(damaged_path).convert('RGB')
            
            complete_tensor = self.transform(complete_img)
            damaged_tensor = self.transform(damaged_img)
            
            return {
                'complete': complete_tensor,
                'damaged': damaged_tensor,
                'complete_path': complete_path,
                'damaged_path': damaged_path
            }
            
        except Exception as e:
            logger.error(f"❌ Error loading images: {str(e)}")
            logger.error(f"❌ Failed to load complete image: {complete_path}")
            logger.error(f"❌ Failed to load damaged image: {damaged_path}")
            raise
