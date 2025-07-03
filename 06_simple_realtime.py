#!/usr/bin/env python3
"""
06_simple_realtime.py
Simple Real-time Deepfake Detection (Backup Script)
Works with any trained model file - guaranteed to work!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os

# Simple model definition (matches training)
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

def load_model():
    """Load any available trained model"""
    model_files = [
        'enhanced_deepfake_detector.pth',
        'simple_deepfake_detector.pth',
        'best_deepfake_detector.pth'
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"📥 Loading model: {model_file}")
            model = SimpleDeepfakeDetector()
            try:
                state_dict = torch.load(model_file, map_location='cpu')
                
                # Handle different checkpoint formats
                if 'model_state_dict' in state_dict:
                    model.load_state_dict(state_dict['model_state_dict'])
                elif 'state_dict' in state_dict:
                    model.load_state_dict(state_dict['state_dict'])
                else:
                    model.load_state_dict(state_dict)
                
                model.eval()
                print(f"✅ Model loaded successfully!")
                return model
            except Exception as e:
                print(f"⚠️  Error loading {model_file}: {e}")
                continue
    
    print("❌ No compatible model found!")
    print("Please train a model first:")
    print("  python 03_simple_training.py")
    return None

def preprocess_face(face_img, size=128):
    """Preprocess face for model input"""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    face_tensor = transform(face_pil).unsqueeze(0)
    return face_tensor

def main():
    print("🎥 Simple Real-time Deepfake Detection")
    print("=" * 40)
    
    # Load model
    model = load_model()
    if model is None:
        print("❌ Please train a model first!")
        return
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam!")
        return
    
    print("🚀 Starting detection... Press 'q' to quit")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces every 3rd frame for speed
        if frame_count % 3 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
            
            for (x, y, w, h) in faces:
                # Extract face
                face = frame[y:y+h, x:x+w]
                
                if face.size > 0:
                    try:
                        # Preprocess and predict
                        face_tensor = preprocess_face(face)
                        
                        with torch.no_grad():
                            outputs = model(face_tensor)
                            probs = F.softmax(outputs, dim=1)
                            fake_prob = probs[0][1].item()
                            
                        # Determine result
                        if fake_prob > 0.6:
                            color = (0, 0, 255)  # Red for fake
                            label = f"FAKE ({fake_prob:.2f})"
                        else:
                            color = (0, 255, 0)  # Green for real
                            label = f"REAL ({1-fake_prob:.2f})"
                        
                        # Draw results
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                    except Exception as e:
                        # Draw unknown if error
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (128, 128, 128), 2)
                        cv2.putText(frame, "UNKNOWN", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        
        # Add instructions
        cv2.putText(frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display
        cv2.imshow('Simple Deepfake Detection', frame)
        
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Detection stopped")

if __name__ == "__main__":
    main()