#!/usr/bin/env python3
"""
05_realtime_detection.py
Enhanced Deepfake Detection System with Real-time Processing
Features: Real-time webcam detection, improved model, better preprocessing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import os
import argparse
import time
import warnings
warnings.filterwarnings('ignore')

# Configuration
class Config:
    # Model settings
    IMAGE_SIZE = 128
    CONFIDENCE_THRESHOLD = 0.6
    FRAME_SKIP = 3  # Process every 3rd frame for speed
    
    # Model paths
    MODEL_PATH = 'enhanced_deepfake_detector.pth'
    FALLBACK_PATHS = [
        'simple_deepfake_detector.pth',
        'best_deepfake_detector.pth',
        'checkpoints/best_model.pth'
    ]

config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

# Enhanced CNN Model
class EnhancedDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(EnhancedDeepfakeDetector, self).__init__()
        
        # Feature extraction
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
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Simple CNN Model (fallback)
class SimpleDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleDeepfakeDetector, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
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

# Face Detection and Preprocessing
class FaceProcessor:
    def __init__(self):
        # Load OpenCV face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
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
            face_pil = Image.fromarray(face_rgb)
            face_tensor = self.transform(face_pil)
            return face_tensor.unsqueeze(0)  # Add batch dimension
        except:
            return None

# Model Loader
def load_model():
    """Load the trained deepfake detection model"""
    
    # Try to load enhanced model first
    model_paths = [config.MODEL_PATH] + config.FALLBACK_PATHS
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            print(f"📥 Attempting to load: {model_path}")
            
            # Try enhanced model first
            try:
                model = EnhancedDeepfakeDetector().to(device)
                
                if model_path.endswith('.pth'):
                    # Load state dict
                    state_dict = torch.load(model_path, map_location=device)
                    if 'model_state_dict' in state_dict:
                        model.load_state_dict(state_dict['model_state_dict'])
                    else:
                        model.load_state_dict(state_dict)
                else:
                    # Load full model
                    model = torch.load(model_path, map_location=device)
                
                model.eval()
                print(f"✅ Enhanced model loaded from: {model_path}")
                return model
                
            except Exception as e:
                print(f"⚠️  Enhanced model failed: {e}")
                
                # Try simple model
                try:
                    model = SimpleDeepfakeDetector().to(device)
                    state_dict = torch.load(model_path, map_location=device)
                    if 'model_state_dict' in state_dict:
                        model.load_state_dict(state_dict['model_state_dict'])
                    else:
                        model.load_state_dict(state_dict)
                    
                    model.eval()
                    print(f"✅ Simple model loaded from: {model_path}")
                    return model
                    
                except Exception as e2:
                    print(f"⚠️  Simple model also failed: {e2}")
                    continue
    
    print("❌ No compatible model found!")
    print("Please train a model first using:")
    print("  python 03_simple_training.py")
    print("  or")
    print("  python 04_advanced_training.py")
    return None

# Real-time Detection System
class RealTimeDetector:
    def __init__(self):
        self.model = load_model()
        if self.model is None:
            return
            
        self.face_processor = FaceProcessor()
        
    def predict_frame(self, frame):
        """Process frame and return predictions for detected faces"""
        if self.model is None:
            return []
            
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
                    fake_prob = probabilities[0][1].item()
                    
                    results.append({
                        'bbox': face_coords,
                        'is_fake': is_fake,
                        'confidence': confidence_score,
                        'fake_prob': fake_prob
                    })
        
        return results
    
    def run_webcam_detection(self):
        """Run real-time detection on webcam feed"""
        if self.model is None:
            print("❌ No model loaded! Cannot start detection.")
            return
            
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open webcam!")
            return
            
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

def test_single_image(image_path):
    """Test deepfake detection on a single image"""
    model = load_model()
    if model is None:
        return
    
    face_processor = FaceProcessor()
    
    # Load and process image
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ Could not load image: {image_path}")
            return
        
        print(f"🔍 Testing image: {image_path}")
        
        # Detect faces
        faces = face_processor.detect_faces(frame)
        
        if len(faces) == 0:
            print("❌ No faces detected in the image!")
            return
        
        print(f"👤 Found {len(faces)} face(s)")
        
        # Process each face
        for i, face_coords in enumerate(faces):
            face_tensor = face_processor.extract_face(frame, face_coords)
            
            if face_tensor is not None:
                with torch.no_grad():
                    face_tensor = face_tensor.to(device)
                    outputs = model(face_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    fake_prob = probabilities[0][1].item()
                    
                # Determine result
                if fake_prob > 0.5:
                    result = "FAKE"
                    confidence = fake_prob
                else:
                    result = "REAL"
                    confidence = 1 - fake_prob
                
                print(f"👤 Face {i+1}:")
                print(f"   🎯 Prediction: {result}")
                print(f"   📊 Confidence: {confidence:.3f}")
                print(f"   🔢 Fake probability: {fake_prob:.3f}")
                
                # Draw results on image
                x, y, w, h = face_coords
                color = (0, 0, 255) if result == "FAKE" else (0, 255, 0)
                label = f"{result} ({confidence:.2f})"
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Save result image
        result_path = f"result_{os.path.basename(image_path)}"
        cv2.imwrite(result_path, frame)
        print(f"💾 Result saved as: {result_path}")
        
        # Display image
        cv2.imshow('Deepfake Detection Result', frame)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Enhanced Deepfake Detection System')
    parser.add_argument('--mode', type=str, choices=['realtime', 'test'], 
                       default='realtime', help='Mode to run')
    parser.add_argument('--image', type=str, help='Path to test image (for test mode)')
    
    args = parser.parse_args()
    
    print("🕵️  Enhanced Deepfake Detection System")
    print("=" * 50)
    
    if args.mode == 'realtime':
        print("🎥 Starting real-time webcam detection...")
        detector = RealTimeDetector()
        if detector.model is not None:
            detector.run_webcam_detection()
        
    elif args.mode == 'test':
        if args.image:
            print(f"🔍 Testing single image: {args.image}")
            test_single_image(args.image)
        else:
            print("❌ Please provide an image path with --image flag")
            print("Example: python 05_realtime_detection.py --mode test --image sample.jpg")
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()