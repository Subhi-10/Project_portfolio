import os
import torch
from datetime import datetime

class Config:
    # Base directory - Now correctly points to "Artifact_restoration"
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # GPU settings
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data parameters
    IMAGE_SIZE = 128
    CHANNELS = 3
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    # Training parameters
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.0001
    BETA1 = 0.5
    BETA2 = 0.999

    # Loss weights
    LAMBDA_PIXEL = 100.0
    LAMBDA_STYLE = 10.0
    LAMBDA_PERCEPTUAL = 5.0
    LAMBDA_ADV = 1.0

    # Model parameters
    GEN_FEATURES = 64
    DISC_FEATURES = 64
    
    NUM_DISPLAY_SAMPLES = 5  # Or any number you prefer
    SAMPLES_PER_WINDOW = 1   # Or any number you prefer

    # Paths - FIXED to correctly point to the actual "data" directory
    DATA_DIR = os.path.join(BASE_DIR, "data")
    COMPLETE_DIR = r"E:\Desktop\Artifact_restoration\data\complete_realistic"
    DAMAGED_DIR = r"E:\Desktop\Artifact_restoration\data\damaged_realistic"

    # Results directory
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    MODEL_VERSION = f"gated_gan_{TIMESTAMP}"
    RESULTS_DIR = os.path.join(BASE_DIR, "results", MODEL_VERSION)
    SAVE_MODEL_DIR = os.path.join(BASE_DIR, "models", MODEL_VERSION)

    # Training settings
    SAVE_INTERVAL = 5
    VALIDATION_INTERVAL = 1

    def __init__(self):
        # Create necessary directories
        os.makedirs(self.RESULTS_DIR, exist_ok=True)
        os.makedirs(self.SAVE_MODEL_DIR, exist_ok=True)

        # ✅ Fix: Check in the correct "data" directory (not inside "src/")
        if not os.path.exists(self.COMPLETE_DIR) or not os.path.exists(self.DAMAGED_DIR):
            raise FileNotFoundError(
                f"❌ Data directories not found! Ensure these paths exist:\n"
                f"✅ Complete images: {self.COMPLETE_DIR}\n"
                f"✅ Damaged images: {self.DAMAGED_DIR}"
            )

        # Verify images exist
        complete_images = [f for f in os.listdir(self.COMPLETE_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        damaged_images = [f for f in os.listdir(self.DAMAGED_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        if not complete_images or not damaged_images:
            raise FileNotFoundError(
                f"❌ Data directories are empty! Please add images to:\n"
                f"✅ Complete images: {self.COMPLETE_DIR} ({len(complete_images)} found)\n"
                f"✅ Damaged images: {self.DAMAGED_DIR} ({len(damaged_images)} found)"
            )

        print(f"✅ Found {len(complete_images)} complete images and {len(damaged_images)} damaged images")

