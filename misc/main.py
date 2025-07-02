#!/usr/bin/env python3
"""
Ultra-Fast Deepfake Detection Trainer
Gets you results in under 5 minutes on CPU!
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from PIL import Image
import os
import random
from tqdm import tqdm
import time

# ULTRA-FAST SETTINGS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

# OPTIMIZED HYPERPARAMETERS FOR SPEED
BATCH_SIZE = 32          # Larger batches for efficiency
LEARNING_RATE = 0.001    # Higher LR for faster convergence
NUM_EPOCHS = 3           # Just 3 epochs for demo
IMAGE_SIZE = 64          # Much smaller images (16x faster than 224)
MAX_TRAIN_SAMPLES = 1000 # Only 1000 training samples
MAX_VAL_SAMPLES = 200    # Only 200 validation samples
MAX_TEST_SAMPLES = 200   # Only 200 test samples

print(f"⚡ SPEED SETTINGS:")
print(f"   Image size: {IMAGE_SIZE}x{IMAGE_SIZE} (16x faster than 224x224)")
print(f"   Training samples: {MAX_TRAIN_SAMPLES}")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Expected time: 3-5 minutes on CPU")

# SUPER SIMPLE TRANSFORMS
transform_train = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# SIMPLE DATASET CLASS
class FastDeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None, max_samples=None):
        self.transform = transform
        self.data = []
        self.labels = []
        
        # Load real images
        if os.path.exists(real_dir):
            real_files = [f for f in os.listdir(real_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if max_samples:
                real_files = real_files[:max_samples//2]
            
            for filename in real_files:
                self.data.append(os.path.join(real_dir, filename))
                self.labels.append(0)  # 0 for real
        
        # Load fake images
        if os.path.exists(fake_dir):
            fake_files = [f for f in os.listdir(fake_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if max_samples:
                fake_files = fake_files[:max_samples//2]
                
            for filename in fake_files:
                self.data.append(os.path.join(fake_dir, filename))
                self.labels.append(1)  # 1 for fake
        
        print(f"   Dataset: {len(self.data)} images ({sum(self.labels)} fake, {len(self.labels) - sum(self.labels)} real)")
    
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
        except:
            # Return a dummy image if loading fails
            dummy_image = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
            return dummy_image, label

# TINY BUT EFFECTIVE MODEL
class TinyDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(TinyDeepfakeDetector, self).__init__()
        
        # Very simple but effective architecture
        self.features = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2: 32x32 -> 16x16  
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Global pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# FAST TRAINING FUNCTION
def fast_train(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    print(f"\n🏃‍♂️ Starting FAST training...")
    
    train_accs = []
    val_accs = []
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            acc = 100 * correct / total
            pbar.set_postfix({'Loss': f'{loss.item():.3f}', 'Acc': f'{acc:.1f}%'})
        
        train_acc = 100 * correct / total
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_accs.append(val_acc)
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}% (Time: {elapsed:.1f}s)")
    
    total_time = time.time() - start_time
    print(f"\n✅ Training completed in {total_time:.1f} seconds!")
    
    return train_accs, val_accs

# FAST EVALUATION
def fast_evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    return accuracy, all_preds, all_labels

# SETUP FAST DATASET
def setup_fast_celeb_df():
    print("📊 Setting up FAST Celeb-DF dataset...")
    
    splits_dir = "data/celeb_df/splits"
    if not os.path.exists(splits_dir):
        print("❌ Dataset not found! Run download_celeb_df.py first")
        return None, None, None
    
    # Create SMALL datasets
    train_dataset = FastDeepfakeDataset(
        real_dir='data/celeb_df/splits/train/real',
        fake_dir='data/celeb_df/splits/train/fake',
        transform=transform_train,
        max_samples=MAX_TRAIN_SAMPLES
    )
    
    val_dataset = FastDeepfakeDataset(
        real_dir='data/celeb_df/splits/val/real',
        fake_dir='data/celeb_df/splits/val/fake',
        transform=transform_test,
        max_samples=MAX_VAL_SAMPLES
    )
    
    test_dataset = FastDeepfakeDataset(
        real_dir='data/celeb_df/splits/test/real',
        fake_dir='data/celeb_df/splits/test/fake',
        transform=transform_test,
        max_samples=MAX_TEST_SAMPLES
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader

# VISUALIZATION
def plot_fast_results(train_accs, val_accs):
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    epochs = range(1, len(train_accs) + 1)
    plt.plot(epochs, train_accs, 'b-', label='Training')
    plt.plot(epochs, val_accs, 'r-', label='Validation')
    plt.title('Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    final_train = train_accs[-1]
    final_val = val_accs[-1]
    plt.bar(['Train', 'Validation'], [final_train, final_val], color=['blue', 'red'], alpha=0.7)
    plt.title('Final Accuracies')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    
    # Add text on bars
    plt.text(0, final_train + 1, f'{final_train:.1f}%', ha='center')
    plt.text(1, final_val + 1, f'{final_val:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# MAIN FUNCTION
def main():
    print("🚀 ULTRA-FAST Celeb-DF Deepfake Detection")
    print("=" * 50)
    
    # Setup data
    train_loader, val_loader, test_loader = setup_fast_celeb_df()
    if train_loader is None:
        return
    
    # Initialize tiny model
    model = TinyDeepfakeDetector().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model parameters: {num_params:,} (much smaller!)")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train fast
    train_accs, val_accs = fast_train(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)
    
    # Test
    print("\n🧪 Final testing...")
    test_acc, predictions, true_labels = fast_evaluate(model, test_loader)
    
    # Results
    print(f"\n🎉 RESULTS:")
    print(f"   Final Training Accuracy: {train_accs[-1]:.1f}%")
    print(f"   Final Validation Accuracy: {val_accs[-1]:.1f}%")
    print(f"   Test Accuracy: {test_acc:.1f}%")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(true_labels, predictions, target_names=['Real', 'Fake']))
    
    # Save model
    torch.save(model.state_dict(), 'fast_deepfake_detector.pth')
    print(f"\n💾 Model saved as 'fast_deepfake_detector.pth'")
    
    # Plot results
    print("\n📈 Generating plots...")
    plot_fast_results(train_accs, val_accs)
    plot_confusion_matrix(true_labels, predictions)
    
    # Performance analysis
    if test_acc > 70:
        print(f"🎯 Great! {test_acc:.1f}% accuracy is excellent for a 5-minute training!")
    elif test_acc > 60:
        print(f"👍 Good! {test_acc:.1f}% accuracy in just {NUM_EPOCHS} epochs!")
    else:
        print(f"📚 {test_acc:.1f}% accuracy - try running with more epochs or samples for better results")
    
    print(f"\n🚀 Next steps to improve:")
    print(f"   - Increase MAX_TRAIN_SAMPLES to 5000+")
    print(f"   - Increase NUM_EPOCHS to 10+") 
    print(f"   - Increase IMAGE_SIZE to 112 or 224")
    print(f"   - Use GPU if available")

if __name__ == "__main__":
    main()