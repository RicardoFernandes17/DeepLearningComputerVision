#!/usr/bin/env python3
"""
Complete Training Pipeline for Deepfake Detection
Features: Comprehensive logging, model optimization, advanced metrics, checkpoint management
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
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

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import the enhanced model - handle if it doesn't exist
try:
    from enhanced_deepfake_detector import EnhancedDeepfakeDetector, Config
except ImportError:
    print("⚠️  enhanced_deepfake_detector.py not found. Using standalone model definition.")
    
    # Standalone model definition
    class EnhancedDeepfakeDetector(nn.Module):
        def __init__(self, num_classes=2, dropout_rate=0.5):
            super(EnhancedDeepfakeDetector, self).__init__()
            
            # Feature extraction with residual connections
            self.features = nn.Sequential(
                # Block 1: 128x128 -> 64x64
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.1),
                
                # Block 2: 64x64 -> 32x32
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.1),
                
                # Block 3: 32x32 -> 16x16
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.2),
                
                # Block 4: 16x16 -> 8x8
                nn.Conv2d(256, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                # Global pooling
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            
            # Enhanced classifier with attention
            self.classifier = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(128, num_classes)
            )
            
            # Initialize weights
            self._initialize_weights()
        
        def _initialize_weights(self):
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
            x = self.features(x)
            x = self.classifier(x)
            return x

warnings.filterwarnings('ignore')

# Setup logging
def setup_logging(log_dir="logs"):
    """Setup comprehensive logging"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# Enhanced configuration with hyperparameter optimization
class TrainingConfig:
    # Model parameters
    MODEL_NAME = "EnhancedDeepfakeDetector"
    NUM_CLASSES = 2
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 5
    WEIGHT_DECAY = 1e-4
    GRADIENT_CLIP_NORM = 1.0
    
    # Data parameters
    IMAGE_SIZE = 64
    MAX_TRAIN_SAMPLES = 3000
    MAX_VAL_SAMPLES = 600
    MAX_TEST_SAMPLES = 600
    
    # Optimization parameters
    EARLY_STOPPING_PATIENCE = 5
    LR_SCHEDULER_PATIENCE = 3
    LR_SCHEDULER_FACTOR = 0.5
    MIN_LR = 1e-6
    
    # Augmentation parameters
    AUGMENTATION_PROB = 0.8
    MIXUP_ALPHA = 0.2
    CUTMIX_ALPHA = 1.0
    
    # Hardware settings
    USE_MIXED_PRECISION = True
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Paths
    CHECKPOINT_DIR = "checkpoints"
    RESULTS_DIR = "results"
    MODEL_PATH = "best_deepfake_detector.pth"
    DATA_PATH = "data/celeb_df/splits"
    
    def __init__(self):
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.RESULTS_DIR, exist_ok=True)

# Advanced data augmentation
class AdvancedAugmentation:
    @staticmethod
    def get_train_transforms(image_size=128):
        return A.Compose([
            A.Resize(image_size, image_size),
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=0.2),
                A.RandomRotate90(p=0.5),
            ], p=0.5),
            
            # Color augmentations
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
            ], p=0.6),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                A.MultiplicativeNoise(multiplier=[0.9, 1.1], per_channel=True, p=1.0),
            ], p=0.3),
            
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.2),
            
            # Geometric transformations
            A.OneOf([
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=1.0),
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1.0),
                A.GridDistortion(p=1.0),
                A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=1.0),
            ], p=0.3),
            
            # Quality degradation
            A.OneOf([
                A.ImageCompression(quality_lower=75, quality_upper=100, p=1.0),
                A.Downscale(scale_min=0.75, scale_max=0.95, p=1.0),
            ], p=0.2),
            
            # Cutout and masking
            A.OneOf([
                A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=1.0),
                A.GridDropout(ratio=0.2, p=1.0),
            ], p=0.2),
            
            # Normalization
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    @staticmethod
    def get_val_transforms(image_size=128):
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

# Enhanced dataset with better error handling
class OptimizedDeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None, max_samples=None, balance_classes=True):
        self.transform = transform
        self.data = []
        self.labels = []
        
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
        
        print(f"Dataset loaded: {len(self.data)} images ({sum(self.labels)} fake, {len(self.labels) - sum(self.labels)} real)")
    
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
            # Load with OpenCV and convert to RGB
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Could not load image: {img_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            else:
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                
            return image, label
            
        except Exception as e:
            # Return a black image if loading fails
            print(f"Warning: Error loading {img_path}: {e}")
            dummy_image = torch.zeros(3, 128, 128)
            return dummy_image, label

# Mixup and CutMix augmentation
class MixupCutmix:
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
    
    def __call__(self, batch):
        if np.random.rand() > self.prob:
            return batch
        
        images, labels = batch
        batch_size = images.size(0)
        
        if np.random.rand() < 0.5:
            # Mixup
            indices = torch.randperm(batch_size)
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            
            mixed_images = lam * images + (1 - lam) * images[indices]
            
            return mixed_images, labels, labels[indices], lam
        else:
            # CutMix
            indices = torch.randperm(batch_size)
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
            
            mixed_images = images.clone()
            mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images[indices, :, bbx1:bbx2, bby1:bby2]
            
            # Adjust lambda to actual pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            
            return mixed_images, labels, labels[indices], lam
    
    def _rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2

# Training metrics tracker
class MetricsTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        self.epoch_times = []
        
        self.best_val_acc = 0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
    
    def update(self, train_loss, train_acc, val_loss, val_acc, lr, epoch_time):
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)
        
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch = len(self.val_accuracies) - 1
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
    
    def save_history(self, filepath):
        history = {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times,
            'best_val_accuracy': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'total_epochs': len(self.train_losses),
            'final_train_accuracy': self.train_accuracies[-1] if self.train_accuracies else 0,
            'final_val_accuracy': self.val_accuracies[-1] if self.val_accuracies else 0
        }
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)

# Advanced trainer class
class AdvancedTrainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics_tracker = MetricsTracker()
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.USE_MIXED_PRECISION and torch.cuda.is_available() else None
        
        # Mixup/Cutmix
        self.mixup_cutmix = MixupCutmix(config.MIXUP_ALPHA, config.CUTMIX_ALPHA)
        
        self.logger.info(f"Trainer initialized on device: {self.device}")
    
    def setup_model(self):
        """Initialize model with proper weight initialization"""
        model = EnhancedDeepfakeDetector(num_classes=self.config.NUM_CLASSES)
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
        
        return model
    
    def setup_data_loaders(self):
        """Setup optimized data loaders with class balancing"""
        train_dataset = OptimizedDeepfakeDataset(
            real_dir=os.path.join(self.config.DATA_PATH, 'train/real'),
            fake_dir=os.path.join(self.config.DATA_PATH, 'train/fake'),
            transform=AdvancedAugmentation.get_train_transforms(self.config.IMAGE_SIZE),
            max_samples=self.config.MAX_TRAIN_SAMPLES,
            balance_classes=True
        )
        
        val_dataset = OptimizedDeepfakeDataset(
            real_dir=os.path.join(self.config.DATA_PATH, 'val/real'),
            fake_dir=os.path.join(self.config.DATA_PATH, 'val/fake'),
            transform=AdvancedAugmentation.get_val_transforms(self.config.IMAGE_SIZE),
            max_samples=self.config.MAX_VAL_SAMPLES,
            balance_classes=True
        )
        
        test_dataset = OptimizedDeepfakeDataset(
            real_dir=os.path.join(self.config.DATA_PATH, 'test/real'),
            fake_dir=os.path.join(self.config.DATA_PATH, 'test/fake'),
            transform=AdvancedAugmentation.get_val_transforms(self.config.IMAGE_SIZE),
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
        
        self.logger.info(f"Data loaders created: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)} batches")
        
        return train_loader, val_loader, test_loader, class_weights
    
    def setup_training_components(self, model, class_weights):
        """Setup optimizer, scheduler, and loss function"""
        # Weighted loss function
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        
        # Optimizer with weight decay
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min',
            patience=self.config.LR_SCHEDULER_PATIENCE,
            factor=self.config.LR_SCHEDULER_FACTOR,
            min_lr=self.config.MIN_LR
        )
        
        return criterion, optimizer, scheduler
    
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """Train for one epoch with advanced techniques"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.NUM_EPOCHS}")
        
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
        """Validate model performance"""
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
    
    def save_checkpoint(self, model, optimizer, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': self.metrics_tracker.best_val_acc,
            'config': self.config.__dict__
        }
        
        # Regular checkpoint
        checkpoint_path = os.path.join(self.config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Best model
        if is_best:
            best_path = os.path.join(self.config.CHECKPOINT_DIR, 'best_model.pth')
            torch.save(checkpoint, best_path)
            torch.save(model.state_dict(), self.config.MODEL_PATH)
            self.logger.info(f"New best model saved with validation accuracy: {self.metrics_tracker.best_val_acc:.2f}%")
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting advanced training pipeline")
        
        # Setup components
        model = self.setup_model()
        train_loader, val_loader, test_loader, class_weights = self.setup_data_loaders()
        criterion, optimizer, scheduler = self.setup_training_components(model, class_weights)
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.config.NUM_EPOCHS):
            epoch_start_time = time.time()
            
            # Train and validate
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer, epoch)
            val_loss, val_acc = self.validate_epoch(model, val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Update metrics
            self.metrics_tracker.update(train_loss, train_acc, val_loss, val_acc, current_lr, epoch_time)
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch+1:2d}/{self.config.NUM_EPOCHS} | "
                f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}% | "
                f"Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}% | "
                f"LR={current_lr:.6f} | Time={epoch_time:.1f}s"
            )
            
            # Save checkpoint
            is_best = val_acc > self.metrics_tracker.best_val_acc
            if epoch % 5 == 0 or is_best:  # Save every 5 epochs or if best
                self.save_checkpoint(model, optimizer, epoch, is_best)
            
            # Early stopping
            if self.metrics_tracker.epochs_without_improvement >= self.config.EARLY_STOPPING_PATIENCE:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.1f} seconds")
        
        # Save training history
        history_path = os.path.join(self.config.RESULTS_DIR, 'training_history.json')
        self.metrics_tracker.save_history(history_path)
        
        # Final evaluation
        self.logger.info("Starting final evaluation...")
        self.evaluate_model(model, test_loader)
        
        return model
    
    def evaluate_model(self, model, test_loader):
        """Comprehensive model evaluation"""
        # Load best model
        if os.path.exists(self.config.MODEL_PATH):
            model.load_state_dict(torch.load(self.config.MODEL_PATH, map_location=self.device))
            self.logger.info("Loaded best model for evaluation")
        
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating"):
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
        self.logger.info("FINAL EVALUATION RESULTS")
        self.logger.info("="*60)
        self.logger.info(f"Test Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        self.logger.info(f"Precision:      {precision:.4f}")
        self.logger.info(f"Recall:         {recall:.4f}")
        self.logger.info(f"F1-Score:       {f1:.4f}")
        self.logger.info(f"AUC-ROC:        {auc:.4f}")
        
        # Detailed classification report
        self.logger.info("\nDetailed Classification Report:")
        self.logger.info("\n" + classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))
        
        # Save evaluation results
        results = {
            'test_accuracy': float(accuracy),
            'test_precision': float(precision),
            'test_recall': float(recall),
            'test_f1_score': float(f1),
            'test_auc_roc': float(auc),
            'best_val_accuracy': self.metrics_tracker.best_val_acc,
            'total_epochs': len(self.metrics_tracker.train_losses),
            'predictions': all_preds,
            'true_labels': all_labels,
            'probabilities': all_probs
        }
        
        results_path = os.path.join(self.config.RESULTS_DIR, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in results.items()}
            json.dump(json_results, f, indent=2)
        
        # Generate visualizations
        self.generate_evaluation_plots(all_labels, all_preds, all_probs)
        
        return results
    
    def generate_evaluation_plots(self, true_labels, predictions, probabilities):
        """Generate comprehensive evaluation plots"""
        # Set style - handle different matplotlib versions
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                pass  # Use default style if seaborn not available
                
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Training History
        ax1 = plt.subplot(3, 4, 1)
        epochs = range(1, len(self.metrics_tracker.train_losses) + 1)
        plt.plot(epochs, self.metrics_tracker.train_losses, 'b-', label='Train Loss', linewidth=2)
        plt.plot(epochs, self.metrics_tracker.val_losses, 'r-', label='Val Loss', linewidth=2)
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Accuracy History
        ax2 = plt.subplot(3, 4, 2)
        plt.plot(epochs, self.metrics_tracker.train_accuracies, 'b-', label='Train Acc', linewidth=2)
        plt.plot(epochs, self.metrics_tracker.val_accuracies, 'r-', label='Val Acc', linewidth=2)
        plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Learning Rate Schedule
        ax3 = plt.subplot(3, 4, 3)
        plt.plot(epochs, self.metrics_tracker.learning_rates, 'g-', linewidth=2)
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # 4. Epoch Times
        ax4 = plt.subplot(3, 4, 4)
        plt.plot(epochs, self.metrics_tracker.epoch_times, 'orange', linewidth=2)
        plt.title('Training Time per Epoch', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.grid(True, alpha=0.3)
        
        # 5. Confusion Matrix
        ax5 = plt.subplot(3, 4, 5)
        cm = confusion_matrix(true_labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # 6. ROC Curve
        ax6 = plt.subplot(3, 4, 6)
        fpr, tpr, _ = roc_curve(true_labels, probabilities)
        auc = roc_auc_score(true_labels, probabilities)
        plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=1, alpha=0.7)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. Prediction Distribution
        ax7 = plt.subplot(3, 4, 7)
        real_probs = [probabilities[i] for i in range(len(probabilities)) if true_labels[i] == 0]
        fake_probs = [probabilities[i] for i in range(len(probabilities)) if true_labels[i] == 1]
        
        plt.hist(real_probs, bins=30, alpha=0.7, label='Real Images', color='blue', density=True)
        plt.hist(fake_probs, bins=30, alpha=0.7, label='Fake Images', color='red', density=True)
        plt.title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Fake Probability')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. Class-wise Performance
        ax8 = plt.subplot(3, 4, 8)
        metrics_per_class = classification_report(true_labels, predictions, output_dict=True)
        classes = ['Real', 'Fake']
        precision_scores = [metrics_per_class['0']['precision'], metrics_per_class['1']['precision']]
        recall_scores = [metrics_per_class['0']['recall'], metrics_per_class['1']['recall']]
        f1_scores = [metrics_per_class['0']['f1-score'], metrics_per_class['1']['f1-score']]
        
        x = np.arange(len(classes))
        width = 0.25
        
        plt.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        plt.bar(x, recall_scores, width, label='Recall', alpha=0.8)
        plt.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        plt.title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.xticks(x, classes)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 9. Training Progress Summary
        ax9 = plt.subplot(3, 4, 9)
        final_metrics = ['Train Acc', 'Val Acc', 'Test Acc', 'Test F1']
        final_values = [
            self.metrics_tracker.train_accuracies[-1],
            self.metrics_tracker.val_accuracies[-1],
            accuracy_score(true_labels, predictions) * 100,
            f1_score(true_labels, predictions, average='weighted') * 100
        ]
        
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        bars = plt.bar(final_metrics, final_values, color=colors, alpha=0.8)
        plt.title('Final Performance Summary', fontsize=14, fontweight='bold')
        plt.ylabel('Score (%)')
        plt.ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, final_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 10. Loss Convergence Analysis
        ax10 = plt.subplot(3, 4, 10)
        train_val_diff = [abs(t - v) for t, v in zip(self.metrics_tracker.train_accuracies, 
                                                    self.metrics_tracker.val_accuracies)]
        plt.plot(epochs, train_val_diff, 'purple', linewidth=2)
        plt.title('Overfitting Analysis', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Train-Val Accuracy Difference (%)')
        plt.grid(True, alpha=0.3)
        
        # 11. Confidence Analysis
        ax11 = plt.subplot(3, 4, 11)
        correct_mask = np.array(predictions) == np.array(true_labels)
        correct_probs = np.array(probabilities)[correct_mask]
        incorrect_probs = np.array(probabilities)[~correct_mask]
        
        # For correct predictions, use max probability (either real or fake)
        correct_confidences = [max(p, 1-p) for p in correct_probs]
        incorrect_confidences = [max(p, 1-p) for p in incorrect_probs]
        
        plt.hist(correct_confidences, bins=20, alpha=0.7, label='Correct', color='green', density=True)
        plt.hist(incorrect_confidences, bins=20, alpha=0.7, label='Incorrect', color='red', density=True)
        plt.title('Confidence Distribution by Correctness', fontsize=14, fontweight='bold')
        plt.xlabel('Confidence')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 12. Model Calibration
        ax12 = plt.subplot(3, 4, 12)
        # Reliability diagram
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        accuracies = []
        confidences = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.array(predictions)[in_bin] == np.array(true_labels)[in_bin]
                avg_accuracy_in_bin = accuracy_in_bin.mean()
                avg_confidence_in_bin = np.array(probabilities)[in_bin].mean()
                
                accuracies.append(avg_accuracy_in_bin)
                confidences.append(avg_confidence_in_bin)
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfect Calibration')
        plt.plot(confidences, accuracies, 'o-', linewidth=2, label='Model Calibration')
        plt.title('Model Calibration Plot', fontsize=14, fontweight='bold')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plots
        plots_path = os.path.join(self.config.RESULTS_DIR, 'evaluation_plots.png')
        plt.savefig(plots_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Evaluation plots saved to {plots_path}")
        
        # Also save individual plots for better viewing
        self.save_individual_plots(true_labels, predictions, probabilities)
        
        plt.show()
    
    def save_individual_plots(self, true_labels, predictions, probabilities):
        """Save individual plots for detailed analysis"""
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(true_labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Deepfake Detection', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        
        # Add percentage annotations
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                percentage = cm[i, j] / total * 100
                plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=12, color='red')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # ROC Curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(true_labels, probabilities)
        auc = roc_auc_score(true_labels, probabilities)
        plt.plot(fpr, tpr, 'b-', linewidth=3, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', linewidth=2, alpha=0.7, label='Random Classifier')
        plt.title('ROC Curve - Deepfake Detection', fontsize=16, fontweight='bold')
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.RESULTS_DIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main training function"""
    # Setup
    config = TrainingConfig()
    logger = setup_logging()
    
    logger.info("="*60)
    logger.info("ENHANCED DEEPFAKE DETECTION TRAINING PIPELINE")
    logger.info("="*60)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # Check data availability
    if not os.path.exists(config.DATA_PATH):
        logger.error(f"Dataset not found at {config.DATA_PATH}")
        logger.error("Please run the dataset download script first!")
        return
    
    # Initialize trainer and start training
    trainer = AdvancedTrainer(config, logger)
    
    try:
        model = trainer.train()
        logger.info("Training completed successfully!")
        
        # Print final summary
        logger.info("="*60)
        logger.info("TRAINING SUMMARY")
        logger.info("="*60)
        logger.info(f"Best validation accuracy: {trainer.metrics_tracker.best_val_acc:.2f}%")
        logger.info(f"Total epochs trained: {len(trainer.metrics_tracker.train_losses)}")
        logger.info(f"Model saved to: {config.MODEL_PATH}")
        logger.info(f"Results saved to: {config.RESULTS_DIR}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()