#!/usr/bin/env python3
"""
03_simple_training.py
Simple Deepfake Detection Training Script
Fast, reliable training with good results - works with basic PyTorch installation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from PIL import Image
import random
from tqdm import tqdm
import time
import json

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

# Simple configuration
class SimpleConfig:
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    IMAGE_SIZE = 128
    MAX_TRAIN_SAMPLES = 2000
    MAX_VAL_SAMPLES = 400
    MAX_TEST_SAMPLES = 400
    MODEL_PATH = 'simple_deepfake_detector.pth'
    DATA_PATH = 'data/celeb_df/splits'

config = SimpleConfig()

# Simple transforms
train_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Simple dataset class
class SimpleDeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None, max_samples=None):
        self.transform = transform
        self.data = []
        self.labels = []
        
        # Load real images
        if os.path.exists(real_dir):
            real_files = [f for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if max_samples:
                real_files = random.sample(real_files, min(len(real_files), max_samples//2))
            
            for filename in real_files:
                self.data.append(os.path.join(real_dir, filename))
                self.labels.append(0)  # 0 for real
        
        # Load fake images
        if os.path.exists(fake_dir):
            fake_files = [f for f in os.listdir(fake_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if max_samples:
                fake_files = random.sample(fake_files, min(len(fake_files), max_samples//2))
                
            for filename in fake_files:
                self.data.append(os.path.join(fake_dir, filename))
                self.labels.append(1)  # 1 for fake
        
        # Shuffle data
        combined = list(zip(self.data, self.labels))
        random.shuffle(combined)
        self.data, self.labels = zip(*combined) if combined else ([], [])
        
        print(f"Dataset loaded: {len(self.data)} images ({sum(self.labels)} fake, {len(self.labels) - sum(self.labels)} real)")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return dummy image
            dummy_image = torch.zeros(3, config.IMAGE_SIZE, config.IMAGE_SIZE)
            return dummy_image, label

# Simple CNN model
class SimpleDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleDeepfakeDetector, self).__init__()
        
        # Convolutional feature extraction
        self.features = nn.Sequential(
            # Block 1: 128x128 -> 64x64
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2: 64x64 -> 32x32
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3: 32x32 -> 16x16
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4: 16x16 -> 8x8
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def setup_data_loaders():
    """Setup data loaders"""
    train_dataset = SimpleDeepfakeDataset(
        real_dir=os.path.join(config.DATA_PATH, 'train/real'),
        fake_dir=os.path.join(config.DATA_PATH, 'train/fake'),
        transform=train_transform,
        max_samples=config.MAX_TRAIN_SAMPLES
    )
    
    val_dataset = SimpleDeepfakeDataset(
        real_dir=os.path.join(config.DATA_PATH, 'val/real'),
        fake_dir=os.path.join(config.DATA_PATH, 'val/fake'),
        transform=test_transform,
        max_samples=config.MAX_VAL_SAMPLES
    )
    
    test_dataset = SimpleDeepfakeDataset(
        real_dir=os.path.join(config.DATA_PATH, 'test/real'),
        fake_dir=os.path.join(config.DATA_PATH, 'test/fake'),
        transform=test_transform,
        max_samples=config.MAX_TEST_SAMPLES
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        acc = 100 * correct / total
        avg_loss = running_loss / (batch_idx + 1)
        pbar.set_postfix({'Loss': f'{avg_loss:.3f}', 'Acc': f'{acc:.1f}%'})
    
    return running_loss / len(train_loader), 100 * correct / total

def validate_epoch(model, val_loader, criterion):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return running_loss / len(val_loader), 100 * correct / total

def evaluate_model(model, test_loader):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, all_preds, all_labels

def plot_results(train_losses, train_accs, val_losses, val_accs):
    """Plot training results"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Final metrics
    plt.subplot(1, 3, 3)
    final_metrics = ['Train Acc', 'Val Acc']
    final_values = [train_accs[-1], val_accs[-1]]
    plt.bar(final_metrics, final_values, color=['blue', 'red'], alpha=0.7)
    plt.title('Final Accuracies')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    
    for i, v in enumerate(final_values):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def save_results(train_losses, train_accs, val_losses, val_accs, test_acc):
    """Save training results"""
    results = {
        'train_losses': train_losses,
        'train_accuracies': train_accs,
        'val_losses': val_losses,
        'val_accuracies': val_accs,
        'test_accuracy': float(test_acc),
        'best_val_accuracy': float(max(val_accs)),
        'final_train_accuracy': float(train_accs[-1]),
        'final_val_accuracy': float(val_accs[-1])
    }
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Save results
    with open('results/simple_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ Results saved to results/simple_training_results.json")

def main():
    """Main training function"""
    print("🚀 Simple Deepfake Detection Training")
    print("=" * 50)
    
    # Check if dataset exists
    if not os.path.exists(config.DATA_PATH):
        print(f"❌ Dataset not found at {config.DATA_PATH}")
        print("Please run the dataset download script first!")
        return
    
    # Setup data
    train_loader, val_loader, test_loader = setup_data_loaders()
    
    if len(train_loader) == 0:
        print("❌ No training data found!")
        return
    
    # Setup model
    model = SimpleDeepfakeDetector().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Training tracking
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0
    
    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("🏋️  Starting training...")
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(config.NUM_EPOCHS):
        # Train and validate
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion)
        
        # Save metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.MODEL_PATH)
            # Also save as enhanced_deepfake_detector.pth for compatibility
            torch.save(model.state_dict(), 'enhanced_deepfake_detector.pth')
            print(f"✅ New best model saved! Val Acc: {val_acc:.2f}%")
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    total_time = time.time() - start_time
    print(f"⏱️  Training completed in {total_time:.1f} seconds")
    
    # Final evaluation
    print("🧪 Final evaluation...")
    model.load_state_dict(torch.load(config.MODEL_PATH))  # Load best model
    test_acc, predictions, true_labels = evaluate_model(model, test_loader)
    
    # Results
    print("\n🎉 FINAL RESULTS:")
    print(f"   Best Training Accuracy: {max(train_accs):.2f}%")
    print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"   Test Accuracy: {test_acc:.2f}%")
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(true_labels, predictions, target_names=['Real', 'Fake']))
    
    # Visualizations
    print("📊 Generating plots...")
    plot_results(train_losses, train_accs, val_losses, val_accs)
    plot_confusion_matrix(true_labels, predictions)
    
    # Save results
    save_results(train_losses, train_accs, val_losses, val_accs, test_acc * 100)
    
    print(f"\n💾 Model saved as: {config.MODEL_PATH}")
    print(f"💾 Compatible copy saved as: enhanced_deepfake_detector.pth")
    print("🎓 Training complete! Ready for testing and demo.")

if __name__ == "__main__":
    main()