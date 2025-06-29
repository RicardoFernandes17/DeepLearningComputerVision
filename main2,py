import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from PIL import Image
import os
import dlib
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
NUM_EPOCHS = 5
IMAGE_SIZE = 112
NUM_CLASSES = 2  # Real vs Fake

# Advanced data preprocessing for deepfake detection
transform_train = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),  # Small rotation to avoid artifacts
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Face extraction utility
def extract_face(image_path, target_size=(224, 224)):
    """Extract face from image using dlib face detector"""
    try:
        # Initialize face detector
        detector = dlib.get_frontal_face_detector()
        
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector(gray)
        
        if len(faces) > 0:
            # Get the largest face
            face = max(faces, key=lambda rect: rect.width() * rect.height())
            
            # Extract face region with some padding
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            padding = int(0.2 * min(w, h))
            
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(img.shape[1], x + w + padding)
            y_end = min(img.shape[0], y + h + padding)
            
            face_img = img[y_start:y_end, x_start:x_end]
            face_img = cv2.resize(face_img, target_size)
            
            return cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    except:
        pass
    return None

# Custom Dataset for Deepfake Detection
class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None, extract_faces=True):
        self.transform = transform
        self.extract_faces = extract_faces
        self.data = []
        self.labels = []
        
        # Load real images
        if os.path.exists(real_dir):
            for filename in os.listdir(real_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.data.append(os.path.join(real_dir, filename))
                    self.labels.append(0)  # 0 for real
        
        # Load fake images
        if os.path.exists(fake_dir):
            for filename in os.listdir(fake_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.data.append(os.path.join(fake_dir, filename))
                    self.labels.append(1)  # 1 for fake
        
        print(f"Dataset loaded: {len(self.data)} images ({sum(self.labels)} fake, {len(self.labels) - sum(self.labels)} real)")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        
        # Load image
        if self.extract_faces:
            image = extract_face(img_path, (IMAGE_SIZE, IMAGE_SIZE))
            if image is None:
                # Fallback to regular image loading
                image = Image.open(img_path).convert('RGB')
        else:
            image = Image.open(img_path).convert('RGB')
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Advanced CNN Architecture for Deepfake Detection
class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepfakeDetector, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Transfer Learning Model for Deepfake Detection
class EfficientNetDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(EfficientNetDeepfakeDetector, self).__init__()
        
        # Use EfficientNet as backbone
        self.backbone = torchvision.models.efficientnet_b0(pretrained=pretrained)
        
        # Modify classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Attention-based Model
class AttentionDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(AttentionDeepfakeDetector, self).__init__()
        
        # Feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Self-attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Feature extraction
        features = self.conv_layers(x)  # [B, 256, H, W]
        
        # Reshape for attention
        B, C, H, W = features.shape
        features_flat = features.view(B, C, H*W).permute(0, 2, 1)  # [B, H*W, C]
        
        # Apply attention
        attended_features, _ = self.attention(features_flat, features_flat, features_flat)
        
        # Reshape back
        attended_features = attended_features.permute(0, 2, 1).view(B, C, H, W)
        
        # Classify
        output = self.classifier(attended_features)
        return output

# Training function with advanced metrics
def train_deepfake_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience=5):
    model.train()
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation phase
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_deepfake_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# Enhanced evaluation function
def evaluate_model(model, test_loader, criterion=None):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            if criterion:
                loss = criterion(outputs, labels)
                test_loss += loss.item()
            
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader) if criterion else 0
    
    return avg_loss, accuracy, all_predictions, all_labels, all_probabilities

# Visualization functions
def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(train_accs, label='Training Accuracy')
    ax2.plot(val_accs, label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes=['Real', 'Fake']):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Prediction function
def predict_deepfake(model, image_path, transform):
    """Predict if an image is a deepfake"""
    model.eval()
    
    # Load and preprocess image
    if os.path.exists(image_path):
        # Try to extract face first
        face_img = extract_face(image_path)
        if face_img is not None:
            image = Image.fromarray(face_img)
        else:
            image = Image.open(image_path).convert('RGB')
    else:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Apply transforms
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    result = {
        'prediction': 'Fake' if predicted.item() == 1 else 'Real',
        'confidence': confidence.item() * 100,
        'real_probability': probabilities[0][0].item() * 100,
        'fake_probability': probabilities[0][1].item() * 100
    }
    
    return result

# Setup function for Celeb-DF dataset
def setup_celeb_df_training():
    """Setup training for Celeb-DF dataset downloaded with our script"""
    
    # Check if the dataset exists
    splits_dir = "data/celeb_df/splits"
    if not os.path.exists(splits_dir):
        print("❌ Celeb-DF dataset not found!")
        print("Please run 'python download_celeb_df.py' first to download and organize the dataset.")
        return None, None, None
    
    print("✅ Found Celeb-DF dataset!")
    
    # Create datasets using the organized structure
    train_dataset = DeepfakeDataset(
        real_dir='data/celeb_df/splits/train/real',
        fake_dir='data/celeb_df/splits/train/fake',
        transform=transform_train,
        extract_faces=False  # Faces already extracted in preprocessed dataset
    )
    
    val_dataset = DeepfakeDataset(
        real_dir='data/celeb_df/splits/val/real',
        fake_dir='data/celeb_df/splits/val/fake',
        transform=transform_test,
        extract_faces=False
    )
    
    test_dataset = DeepfakeDataset(
        real_dir='data/celeb_df/splits/test/real',
        fake_dir='data/celeb_df/splits/test/fake',
        transform=transform_test,
        extract_faces=False
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"📊 Dataset splits loaded:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")
    print(f"   Test:  {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader

# Main execution for Celeb-DF
if __name__ == "__main__":
    print("🚀 Celeb-DF Deepfake Detection Training")
    print("=" * 50)
    
    # Setup data loaders
    train_loader, val_loader, test_loader = setup_celeb_df_training()
    
    if train_loader is None:
        print("\n💡 To get started:")
        print("1. First run: python download_celeb_df.py")
        print("2. Then run this training script")
        exit()
    
    # Initialize model
    print(f"\n🤖 Initializing model on {device}")
    model = DeepfakeDetector().to(device)  # or EfficientNetDeepfakeDetector()
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    
    print(f"📊 Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Start training
    print(f"\n🏃‍♂️ Starting training for {NUM_EPOCHS} epochs...")
    train_losses, val_losses, train_accs, val_accs = train_deepfake_model(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS
    )
    
    # Plot training history
    print("\n📈 Plotting training history...")
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # Final evaluation on test set
    print("\n🧪 Final evaluation on test set...")
    test_loss, test_acc, predictions, true_labels, probabilities = evaluate_model(model, test_loader, criterion)
    
    # Plot confusion matrix
    plot_confusion_matrix(true_labels, predictions)
    
    # Print final results
    print(f"\n🎉 Training Complete!")
    print(f"📊 Final Results:")
    print(f"   Best Validation Accuracy: {max(val_accs):.2f}%")
    print(f"   Test Accuracy: {test_acc:.2f}%")
    print(f"   Test Loss: {test_loss:.4f}")
    
    # Classification report
    print(f"\n📋 Detailed Classification Report:")
    print(classification_report(true_labels, predictions, target_names=['Real', 'Fake']))
    
    # Save final model
    torch.save(model.state_dict(), 'celeb_df_deepfake_detector.pth')
    print(f"\n💾 Model saved as 'celeb_df_deepfake_detector.pth'")
    
    # Example prediction
    print(f"\n🔍 Testing prediction function...")
    if len(test_dataset.data) > 0:
        sample_image = test_dataset.data[0]
        try:
            result = predict_deepfake(model, sample_image, transform_test)
            print(f"Sample prediction: {result['prediction']} (confidence: {result['confidence']:.1f}%)")
        except Exception as e:
            print(f"Prediction test failed: {e}")
    
    print(f"\n✅ All done! Your Celeb-DF deepfake detector is ready to use!")