#!/usr/bin/env python3
"""
Enhanced Deepfake Detection System with Real-time Processing
Features: Real-time webcam detection, improved model, better preprocessing
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from PIL import Image
import cv2
import os
import random
from tqdm import tqdm
import time
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    # Model settings
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 3
    IMAGE_SIZE = 64  # Decreased for faster processing
    MAX_TRAIN_SAMPLES = 2000
    MAX_VAL_SAMPLES = 400
    MAX_TEST_SAMPLES = 400
    
    # Real-time detection settings
    CONFIDENCE_THRESHOLD = 0.6
    FRAME_SKIP = 3  # Process every 3rd frame for speed
    DETECTION_WINDOW_SIZE = 10  # Average over 10 predictions
    
    # Paths
    MODEL_PATH = 'enhanced_deepfake_detector.pth'
    DATA_PATH = 'data/celeb_df/splits'

config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

# Enhanced Data Augmentation with Albumentations
def get_train_transforms():
    return A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.RandomGamma(),
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_test_transforms():
    return A.Compose([
        A.Resize(config.IMAGE_SIZE, config.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# Enhanced Dataset Class
class EnhancedDeepfakeDataset(Dataset):
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
        self.data, self.labels = zip(*combined)
        
        print(f"   Dataset: {len(self.data)} images ({sum(self.labels)} fake, {len(self.labels) - sum(self.labels)} real)")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        
        try:
            # Load image with OpenCV and convert to RGB
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            else:
                # Fallback transform
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
                
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy image if loading fails
            dummy_image = torch.zeros(3, config.IMAGE_SIZE, config.IMAGE_SIZE)
            return dummy_image, label

# Enhanced Model Architecture
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

# Face Detection and Preprocessing
class FaceProcessor:
    def __init__(self):
        # Load OpenCV face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.transform = get_test_transforms()
    
    def detect_faces(self, frame):
        """Detect faces in frame and return bounding boxes"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
        return faces
    
    def extract_face(self, frame, face_coords):
        """Extract and preprocess face for model input"""
        x, y, w, h = face_coords
        
        # Add padding around face
        padding = int(0.2 * max(w, h))
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2*padding)
        h = min(frame.shape[0] - y, h + 2*padding)
        
        # Extract face region
        face = frame[y:y+h, x:x+w]
        
        if face.size == 0:
            return None
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        try:
            augmented = self.transform(image=face_rgb)
            face_tensor = augmented['image']
            return face_tensor.unsqueeze(0)  # Add batch dimension
        except:
            return None

# Real-time Detection System
class RealTimeDetector:
    def __init__(self, model_path=None):
        self.model = EnhancedDeepfakeDetector().to(device)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✅ Model loaded from {model_path}")
        else:
            print("⚠️  No pre-trained model found. Train the model first!")
        
        self.model.eval()
        self.face_processor = FaceProcessor()
        self.prediction_buffer = []
        
    def predict_frame(self, frame):
        """Process frame and return predictions for detected faces"""
        faces = self.face_processor.detect_faces(frame)
        results = []
        
        for face_coords in faces:
            face_tensor = self.face_processor.extract_face(frame, face_coords)
            
            if face_tensor is not None:
                with torch.no_grad():
                    face_tensor = face_tensor.to(device)
                    outputs = self.model(face_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    is_fake = predicted.item() == 1
                    confidence_score = confidence.item()
                    
                    results.append({
                        'bbox': face_coords,
                        'is_fake': is_fake,
                        'confidence': confidence_score,
                        'fake_prob': probabilities[0][1].item()
                    })
        
        return results
    
    def run_webcam_detection(self):
        """Run real-time detection on webcam feed"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        fps_start_time = time.time()
        fps_counter = 0
        current_fps = 0
        
        print("🎥 Starting webcam detection... Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every nth frame for speed
            if frame_count % config.FRAME_SKIP == 0:
                predictions = self.predict_frame(frame)
                
                # Draw results on frame
                for pred in predictions:
                    x, y, w, h = pred['bbox']
                    confidence = pred['confidence']
                    is_fake = pred['is_fake']
                    fake_prob = pred['fake_prob']
                    
                    # Choose color based on prediction
                    if is_fake and confidence > config.CONFIDENCE_THRESHOLD:
                        color = (0, 0, 255)  # Red for fake
                        label = f"FAKE ({fake_prob:.2f})"
                    else:
                        color = (0, 255, 0)  # Green for real
                        label = f"REAL ({1-fake_prob:.2f})"
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Draw label background
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x, y-30), (x + label_size[0], y), color, -1)
                    
                    # Draw label text
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Calculate and display FPS
            fps_counter += 1
            if fps_counter >= 30:
                current_fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            
            # Add FPS and instructions to frame
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Deepfake Detection', frame)
            
            frame_count += 1
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("🛑 Webcam detection stopped")

# Enhanced Training Function
def enhanced_train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    print(f"\n🏃‍♂️ Starting enhanced training...")
    
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            avg_loss = train_loss / (batch_idx + 1)
            acc = 100 * correct / total
            pbar.set_postfix({'Loss': f'{avg_loss:.3f}', 'Acc': f'{acc:.1f}%'})
        
        train_acc = 100 * correct / total
        train_accs.append(train_acc)
        train_losses.append(train_loss / len(train_loader))
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_accs.append(val_acc)
        val_losses.append(val_loss / len(val_loader))
        
        # Learning rate scheduling
        scheduler.step(val_loss / len(val_loader))
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.MODEL_PATH)
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%, Best: {best_val_acc:.1f}% (Time: {elapsed:.1f}s)")
    
    total_time = time.time() - start_time
    print(f"\n✅ Training completed in {total_time:.1f} seconds!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.1f}%")
    
    return train_losses, train_accs, val_losses, val_accs

# Setup Enhanced Dataset
def setup_enhanced_dataset():
    print("📊 Setting up enhanced dataset...")
    
    if not os.path.exists(config.DATA_PATH):
        print("❌ Dataset not found! Run the dataset downloader first")
        return None, None, None
    
    # Create datasets with enhanced transforms
    train_dataset = EnhancedDeepfakeDataset(
        real_dir=os.path.join(config.DATA_PATH, 'train/real'),
        fake_dir=os.path.join(config.DATA_PATH, 'train/fake'),
        transform=get_train_transforms(),
        max_samples=config.MAX_TRAIN_SAMPLES
    )
    
    val_dataset = EnhancedDeepfakeDataset(
        real_dir=os.path.join(config.DATA_PATH, 'val/real'),
        fake_dir=os.path.join(config.DATA_PATH, 'val/fake'),
        transform=get_test_transforms(),
        max_samples=config.MAX_VAL_SAMPLES
    )
    
    test_dataset = EnhancedDeepfakeDataset(
        real_dir=os.path.join(config.DATA_PATH, 'test/real'),
        fake_dir=os.path.join(config.DATA_PATH, 'test/fake'),
        transform=get_test_transforms(),
        max_samples=config.MAX_TEST_SAMPLES
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, test_loader

# Evaluation and Visualization
def enhanced_evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities[:, 1].cpu().numpy())  # Fake probabilities
    
    accuracy = 100 * correct / total
    return accuracy, all_preds, all_labels, all_probs

def plot_training_history(train_losses, train_accs, val_losses, val_accs):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Training and validation loss
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Training and validation accuracy
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # Final accuracies bar chart
    final_train = train_accs[-1]
    final_val = val_accs[-1]
    ax3.bar(['Train', 'Validation'], [final_train, final_val], color=['blue', 'red'], alpha=0.7)
    ax3.set_title('Final Accuracies')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_ylim(0, 100)
    ax3.text(0, final_train + 1, f'{final_train:.1f}%', ha='center')
    ax3.text(1, final_val + 1, f'{final_val:.1f}%', ha='center')
    
    # Learning curves comparison
    ax4.plot(epochs, [abs(t-v) for t, v in zip(train_accs, val_accs)], 'g-', label='Acc Difference')
    ax4.set_title('Train-Val Accuracy Difference')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy Difference (%)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_enhanced_confusion_matrix(y_true, y_pred, class_names=['Real', 'Fake']):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Enhanced Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add percentage annotations
    total = cm.sum()
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            percentage = cm[i, j] / total * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='red')
    
    plt.show()

# Main Functions
def train_model():
    """Train the enhanced deepfake detection model"""
    print("🚀 Enhanced Deepfake Detection Training")
    print("=" * 50)
    
    # Setup data
    train_loader, val_loader, test_loader = setup_enhanced_dataset()
    if train_loader is None:
        return
    
    # Initialize model
    model = EnhancedDeepfakeDetector().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model parameters: {num_params:,}")
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Train model
    train_losses, train_accs, val_losses, val_accs = enhanced_train(
        model, train_loader, val_loader, criterion, optimizer, scheduler, config.NUM_EPOCHS
    )
    
    # Final evaluation
    print("\n🧪 Final testing...")
    model.load_state_dict(torch.load(config.MODEL_PATH))  # Load best model
    test_acc, predictions, true_labels, probabilities = enhanced_evaluate(model, test_loader)
    
    # Results
    print(f"\n🎉 RESULTS:")
    print(f"   Best Training Accuracy: {max(train_accs):.1f}%")
    print(f"   Best Validation Accuracy: {max(val_accs):.1f}%")
    print(f"   Test Accuracy: {test_acc:.1f}%")
    
    # Detailed classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(true_labels, predictions, target_names=['Real', 'Fake']))
    
    # Plot results
    print("\n📈 Generating plots...")
    plot_training_history(train_losses, train_accs, val_losses, val_accs)
    plot_enhanced_confusion_matrix(true_labels, predictions)
    
    return model

def run_realtime_detection():
    """Run real-time deepfake detection"""
    detector = RealTimeDetector(config.MODEL_PATH)
    detector.run_webcam_detection()

def test_single_image(image_path):
    """Test deepfake detection on a single image"""
    if not os.path.exists(config.MODEL_PATH):
        print("❌ No trained model found. Train the model first!")
        return
    
    # Load model
    model = EnhancedDeepfakeDetector().to(device)
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
    model.eval()
    
    # Load and preprocess image
    try:
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        transform = get_test_transforms()
        augmented = transform(image=image_rgb)
        image_tensor = augmented['image'].unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            fake_prob = probabilities[0][1].item()
            
        # Display result
        result = "FAKE" if fake_prob > 0.5 else "REAL"
        confidence = fake_prob if fake_prob > 0.5 else 1 - fake_prob
        
        print(f"🔍 Image: {image_path}")
        print(f"🎯 Prediction: {result} (confidence: {confidence:.3f})")
        print(f"📊 Fake probability: {fake_prob:.3f}")
        
        # Show image with prediction
        plt.figure(figsize=(8, 6))
        plt.imshow(image_rgb)
        plt.title(f"Prediction: {result} (Fake prob: {fake_prob:.3f})")
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Deepfake Detection System')
    parser.add_argument('--mode', type=str, choices=['train', 'realtime', 'test'], 
                       default='train', help='Mode to run')
    parser.add_argument('--image', type=str, help='Path to test image (for test mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("🎓 Starting model training...")
        train_model()
        
    elif args.mode == 'realtime':
        print("🎥 Starting real-time detection...")
        run_realtime_detection()
        
    elif args.mode == 'test':
        if args.image:
            print(f"🔍 Testing single image: {args.image}")
            test_single_image(args.image)
        else:
            print("❌ Please provide an image path with --image flag")
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()