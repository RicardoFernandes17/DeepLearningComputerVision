#!/usr/bin/env python3
"""
09_vit_model.py
Vision Transformer (ViT) Implementation for Deepfake Detection
Implements ViT-based deepfake detector using timm library
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve)

import cv2
import os
import json
import time
import random
from datetime import datetime
from pathlib import Path
import logging
from tqdm import tqdm
import warnings

try:
    import timm
    TIMM_AVAILABLE = True
    print("✅ timm library available")
except ImportError:
    TIMM_AVAILABLE = False
    print("⚠️  timm not available. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "timm"])
    import timm
    TIMM_AVAILABLE = True

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    print("⚠️  Albumentations not available, using basic transforms")
    ALBUMENTATIONS_AVAILABLE = False

warnings.filterwarnings('ignore')

# Configuration for ViT training
class ViTConfig:
    # Model parameters
    MODEL_NAME = "ViTDeepfakeDetector"
    VIT_MODEL = "vit_base_patch16_224"  # Can change to vit_small_patch16_224 for faster training
    NUM_CLASSES = 2
    
    # Training parameters
    BATCH_SIZE = 16  # Smaller batch size for ViT (uses more memory)
    LEARNING_RATE = 1e-4  # Lower LR for pre-trained ViT
    NUM_EPOCHS = 8
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP_NORM = 1.0
    
    # Data parameters
    IMAGE_SIZE = 224  # ViT standard input size
    MAX_TRAIN_SAMPLES = 2000
    MAX_VAL_SAMPLES = 400
    MAX_TEST_SAMPLES = 400
    
    # Optimization parameters
    EARLY_STOPPING_PATIENCE = 4
    LR_SCHEDULER_PATIENCE = 2
    LR_SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-7
    
    # Hardware settings
    USE_MIXED_PRECISION = True
    NUM_WORKERS = 2
    PIN_MEMORY = True
    
    # Paths
    CHECKPOINT_DIR = "checkpoints"
    RESULTS_DIR = "results"
    MODEL_PATH = "vit_deepfake_detector.pth"
    DATA_PATH = "data/celeb_df/splits"
    
    def __init__(self):
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.RESULTS_DIR, exist_ok=True)

# Vision Transformer Model
class ViTDeepfakeDetector(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", num_classes=2, pretrained=True):
        super(ViTDeepfakeDetector, self).__init__()
        
        # Load pre-trained ViT from timm
        self.vit = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=num_classes
        )
        
        # Add dropout for regularization
        if hasattr(self.vit, 'head'):
            # For most ViT models, replace the head
            in_features = self.vit.head.in_features
            self.vit.head = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        
        # Freeze some layers for transfer learning (optional)
        self.freeze_backbone = False
        if self.freeze_backbone:
            for param in self.vit.patch_embed.parameters():
                param.requires_grad = False
            
            # Freeze first few transformer blocks
            for i in range(6):  # Freeze first 6 blocks out of 12
                for param in self.vit.blocks[i].parameters():
                    param.requires_grad = False
    
    def forward(self, x):
        return self.vit(x)
    
    def get_attention_maps(self, x):
        """Extract attention maps for visualization"""
        # This is a simplified version - actual implementation would require
        # modifying the forward pass to return attention weights
        with torch.no_grad():
            # Get features from the model
            features = self.vit.forward_features(x)
            return features

# Enhanced ViT transforms
def get_vit_train_transforms(image_size=224):
    """Get training transforms optimized for ViT"""
    if ALBUMENTATIONS_AVAILABLE:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 30.0)),
                A.ISONoise(),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(blur_limit=3),
                A.MedianBlur(blur_limit=3),
            ], p=0.1),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3),
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_vit_val_transforms(image_size=224):
    """Get validation transforms for ViT"""
    if ALBUMENTATIONS_AVAILABLE:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# Enhanced dataset for ViT (same as before but with ViT transforms)
class ViTDeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None, max_samples=None, balance_classes=True):
        self.transform = transform
        self.data = []
        self.labels = []
        self.albumentations = ALBUMENTATIONS_AVAILABLE and hasattr(transform, 'processors') if transform else False
        
        # Load real images
        real_files = []
        if os.path.exists(real_dir):
            real_files = [f for f in os.listdir(real_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg')) and self._is_valid_image(os.path.join(real_dir, f))]
        
        # Load fake images
        fake_files = []
        if os.path.exists(fake_dir):
            fake_files = [f for f in os.listdir(fake_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg')) and self._is_valid_image(os.path.join(fake_dir, f))]
        
        # Balance classes if requested
        if balance_classes and max_samples:
            samples_per_class = max_samples // 2
            real_files = random.sample(real_files, min(len(real_files), samples_per_class))
            fake_files = random.sample(fake_files, min(len(fake_files), samples_per_class))
        
        # Add real images
        for filename in real_files:
            self.data.append(os.path.join(real_dir, filename))
            self.labels.append(0)  # 0 for real
        
        # Add fake images
        for filename in fake_files:
            self.data.append(os.path.join(fake_dir, filename))
            self.labels.append(1)  # 1 for fake
        
        # Shuffle data
        combined = list(zip(self.data, self.labels))
        random.shuffle(combined)
        self.data, self.labels = zip(*combined) if combined else ([], [])
        
        print(f"ViT Dataset loaded: {len(self.data)} images ({sum(self.labels)} fake, {len(self.labels) - sum(self.labels)} real)")
    
    def _is_valid_image(self, img_path):
        """Check if image is valid and readable"""
        try:
            img = cv2.imread(img_path)
            return img is not None and img.shape[0] > 32 and img.shape[1] > 32
        except:
            return False
    
    def get_class_weights(self):
        """Calculate class weights for balanced training"""
        labels = np.array(self.labels)
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        weights = total_samples / (len(class_counts) * class_counts)
        return torch.FloatTensor(weights)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        
        try:
            if self.albumentations:
                # Load with OpenCV and convert to RGB for albumentations
                image = cv2.imread(img_path)
                if image is None:
                    raise ValueError(f"Could not load image: {img_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Apply albumentations transforms
                if self.transform:
                    augmented = self.transform(image=image)
                    image = augmented['image']
                else:
                    image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            else:
                # Load with PIL for torchvision transforms
                from PIL import Image as PILImage
                image = PILImage.open(img_path).convert('RGB')
                
                if self.transform:
                    image = self.transform(image)
                else:
                    image = transforms.ToTensor()(image)
                
            return image, label
            
        except Exception as e:
            # Return a black image if loading fails
            print(f"Warning: Error loading {img_path}: {e}")
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, label

# ViT Trainer class
class ViTTrainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.USE_MIXED_PRECISION and torch.cuda.is_available() else None
        
        self.logger.info(f"ViT Trainer initialized on device: {self.device}")
    
    def setup_model(self):
        """Initialize ViT model"""
        model = ViTDeepfakeDetector(
            model_name=self.config.VIT_MODEL,
            num_classes=self.config.NUM_CLASSES,
            pretrained=True
        )
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"ViT Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
        
        return model
    
    def setup_data_loaders(self):
        """Setup data loaders for ViT"""
        train_dataset = ViTDeepfakeDataset(
            real_dir=os.path.join(self.config.DATA_PATH, 'train/real'),
            fake_dir=os.path.join(self.config.DATA_PATH, 'train/fake'),
            transform=get_vit_train_transforms(self.config.IMAGE_SIZE),
            max_samples=self.config.MAX_TRAIN_SAMPLES,
            balance_classes=True
        )
        
        val_dataset = ViTDeepfakeDataset(
            real_dir=os.path.join(self.config.DATA_PATH, 'val/real'),
            fake_dir=os.path.join(self.config.DATA_PATH, 'val/fake'),
            transform=get_vit_val_transforms(self.config.IMAGE_SIZE),
            max_samples=self.config.MAX_VAL_SAMPLES,
            balance_classes=True
        )
        
        test_dataset = ViTDeepfakeDataset(
            real_dir=os.path.join(self.config.DATA_PATH, 'test/real'),
            fake_dir=os.path.join(self.config.DATA_PATH, 'test/fake'),
            transform=get_vit_val_transforms(self.config.IMAGE_SIZE),
            max_samples=self.config.MAX_TEST_SAMPLES,
            balance_classes=True
        )
        
        # Class-balanced sampling
        class_weights = train_dataset.get_class_weights()
        sample_weights = [class_weights[label] for label in train_dataset.labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            sampler=sampler,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY
        )
        
        self.logger.info(f"ViT Data loaders created: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)} batches")
        
        return train_loader, val_loader, test_loader, class_weights
    
    def setup_training_components(self, model, class_weights):
        """Setup optimizer, scheduler, and loss function for ViT"""
        # Weighted loss function
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        
        # AdamW optimizer (better for transformers)
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Cosine annealing scheduler (better for ViT)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.config.NUM_EPOCHS,
            eta_min=self.config.MIN_LR
        )
        
        return criterion, optimizer, scheduler
    
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """Train ViT for one epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"ViT Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            # Mixed precision training
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.GRADIENT_CLIP_NORM)
                
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.GRADIENT_CLIP_NORM)
                
                optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            acc = 100 * correct / total
            avg_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({'Loss': f'{avg_loss:.3f}', 'Acc': f'{acc:.1f}%'})
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, model, val_loader, criterion):
        """Validate ViT performance"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """Main ViT training loop"""
        self.logger.info("Starting ViT training pipeline")
        
        # Setup components
        model = self.setup_model()
        train_loader, val_loader, test_loader, class_weights = self.setup_data_loaders()
        criterion, optimizer, scheduler = self.setup_training_components(model, class_weights)
        
        # Training tracking
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        best_val_acc = 0
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.config.NUM_EPOCHS):
            epoch_start_time = time.time()
            
            # Train and validate
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer, epoch)
            val_loss, val_acc = self.validate_epoch(model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Update metrics
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch+1:2d}/{self.config.NUM_EPOCHS} | "
                f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}% | "
                f"Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}% | "
                f"LR={current_lr:.6f} | Time={epoch_time:.1f}s"
            )
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), self.config.MODEL_PATH)
                # Also save with enhanced name for compatibility
                torch.save(model.state_dict(), 'enhanced_vit_deepfake_detector.pth')
                self.logger.info(f"New best ViT model saved with validation accuracy: {best_val_acc:.2f}%")
        
        total_time = time.time() - start_time
        self.logger.info(f"ViT Training completed in {total_time:.1f} seconds")
        
        # Final evaluation
        self.logger.info("Starting ViT final evaluation...")
        test_results = self.evaluate_model(model, test_loader)
        
        # Save results
        results = {
            'model_type': 'ViT',
            'model_name': self.config.VIT_MODEL,
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'best_val_accuracy': float(best_val_acc),
            'test_results': test_results,
            'total_epochs': self.config.NUM_EPOCHS,
            'total_training_time': total_time
        }
        
        results_path = os.path.join(self.config.RESULTS_DIR, 'vit_training_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return model, results
    
    def evaluate_model(self, model, test_loader):
        """Comprehensive ViT model evaluation"""
        # Load best model
        if os.path.exists(self.config.MODEL_PATH):
            model.load_state_dict(torch.load(self.config.MODEL_PATH, map_location=self.device))
            self.logger.info("Loaded best ViT model for evaluation")
        
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating ViT"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = model(images)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probabilities[:, 1].cpu().numpy())  # Fake probabilities
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        auc = roc_auc_score(all_labels, all_probs)
        
        # Log results
        self.logger.info("="*60)
        self.logger.info("ViT FINAL EVALUATION RESULTS")
        self.logger.info("="*60)
        self.logger.info(f"Test Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        self.logger.info(f"Precision:      {precision:.4f}")
        self.logger.info(f"Recall:         {recall:.4f}")
        self.logger.info(f"F1-Score:       {f1:.4f}")
        self.logger.info(f"AUC-ROC:        {auc:.4f}")
        
        # Detailed classification report
        self.logger.info("\nDetailed ViT Classification Report:")
        self.logger.info("\n" + classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))
        
        results = {
            'test_accuracy': float(accuracy),
            'test_precision': float(precision),
            'test_recall': float(recall),
            'test_f1_score': float(f1),
            'test_auc_roc': float(auc),
            'predictions': [int(x) for x in all_preds],
            'true_labels': [int(x) for x in all_labels],
            'probabilities': [float(x) for x in all_probs]
        }
        
        return results

# Setup logging
def setup_logging(log_dir="logs"):
    """Setup comprehensive logging"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"vit_training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def main():
    """Main ViT training function"""
    # Setup
    config = ViTConfig()
    logger = setup_logging()
    
    logger.info("="*60)
    logger.info("VISION TRANSFORMER DEEPFAKE DETECTION TRAINING")
    logger.info("="*60)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
    
    if not TIMM_AVAILABLE:
        logger.error("timm library is required for ViT implementation")
        logger.error("Please install with: pip install timm")
        return
    
    # Check data availability
    if not os.path.exists(config.DATA_PATH):
        logger.error(f"Dataset not found at {config.DATA_PATH}")
        logger.error("Please run the dataset download script first!")
        return
    
    # Initialize trainer and start training
    trainer = ViTTrainer(config, logger)
    
    try:
        model, results = trainer.train()
        logger.info("ViT Training completed successfully!")
        
        # Print final summary
        logger.info("="*60)
        logger.info("ViT TRAINING SUMMARY")
        logger.info("="*60)
        logger.info(f"Best validation accuracy: {results['best_val_accuracy']:.2f}%")
        logger.info(f"Test accuracy: {results['test_results']['test_accuracy']*100:.2f}%")
        logger.info(f"Total epochs trained: {results['total_epochs']}")
        logger.info(f"Model saved to: {config.MODEL_PATH}")
        logger.info(f"Compatible copy saved as: enhanced_vit_deepfake_detector.pth")
        logger.info(f"Results saved to: {config.RESULTS_DIR}")
        
    except KeyboardInterrupt:
        logger.info("ViT Training interrupted by user")
    except Exception as e:
        logger.error(f"ViT Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()