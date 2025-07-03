#!/usr/bin/env python3
"""
10_interpretability.py
Model Interpretability and Grad-CAM Implementation
Provides visualization tools to understand what the model is learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import json
from pathlib import Path
import argparse
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import model architectures
try:
    # Try to import from main scripts
    import sys
    sys.path.append('.')
    from enhanced_deepfake_detector import EnhancedDeepfakeDetector
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False

try:
    import timm
    VIT_AVAILABLE = True
except ImportError:
    VIT_AVAILABLE = False

# Fallback model definitions if imports fail
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

class ViTDeepfakeDetector(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", num_classes=2, pretrained=True):
        super(ViTDeepfakeDetector, self).__init__()
        
        if VIT_AVAILABLE:
            # Load pre-trained ViT from timm
            self.vit = timm.create_model(
                model_name, 
                pretrained=pretrained, 
                num_classes=num_classes
            )
        else:
            # Fallback to CNN if ViT not available
            print("⚠️  ViT not available, using CNN fallback")
            self.vit = SimpleDeepfakeDetector(num_classes)
    
    def forward(self, x):
        return self.vit(x)

# Grad-CAM Implementation
class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find target layer
        target_layer = self._find_target_layer()
        if target_layer is not None:
            hook1 = target_layer.register_forward_hook(forward_hook)
            hook2 = target_layer.register_backward_hook(backward_hook)
            self.hooks.extend([hook1, hook2])
        else:
            print(f"⚠️  Could not find target layer: {self.target_layer_name}")
    
    def _find_target_layer(self):
        """Find the target layer in the model"""
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                return module
        
        # If exact name not found, try partial matching
        for name, module in self.model.named_modules():
            if self.target_layer_name in name:
                print(f"Found partial match: {name}")
                return module
        
        return None
    
    def generate_cam(self, input_tensor, class_idx=None):
        """Generate Grad-CAM heatmap"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        class_score = output[0, class_idx]
        class_score.backward()
        
        if self.gradients is None or self.activations is None:
            print("⚠️  Could not capture gradients and activations")
            return None, class_idx
        
        # Generate CAM
        gradients = self.gradients[0]  # Remove batch dimension
        activations = self.activations[0]  # Remove batch dimension
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.detach().numpy(), class_idx
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()

# Visualization utilities
class InterpretabilityVisualizer:
    def __init__(self, model_path, model_type='cnn'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path):
        """Load the trained model"""
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return None
        
        try:
            if self.model_type == 'vit':
                model = ViTDeepfakeDetector()
            else:
                # Try enhanced model first, fallback to simple
                try:
                    model = EnhancedDeepfakeDetector()
                except:
                    model = SimpleDeepfakeDetector()
            
            # Load state dict
            state_dict = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            else:
                model.load_state_dict(state_dict)
            
            model = model.to(self.device)
            model.eval()
            
            print(f"✅ Model loaded successfully from {model_path}")
            return model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """Load and preprocess image"""
        try:
            # Load image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = image_path
            
            original_image = image.copy()
            
            # Resize
            image = cv2.resize(image, target_size)
            
            # Normalize
            image = image.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean) / std
            
            # Convert to tensor
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
            
            return image_tensor.to(self.device), original_image
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            return None, None
    
    def get_target_layer(self):
        """Get appropriate target layer for Grad-CAM"""
        if self.model_type == 'vit':
            # For ViT, use the last attention layer
            target_layers = ['vit.blocks.11.norm1', 'vit.blocks.11', 'vit.norm']
        else:
            # For CNN, use the last convolutional layer
            target_layers = ['features.12', 'features.11', 'features.10', 'features']
        
        # Find the first available layer
        for layer_name in target_layers:
            for name, _ in self.model.named_modules():
                if layer_name in name:
                    print(f"Using target layer: {name}")
                    return name
        
        # Fallback
        print("⚠️  Using fallback target layer")
        return list(self.model.named_modules())[-2][0]  # Second to last layer
    
    def generate_gradcam(self, image_path, save_path=None):
        """Generate and visualize Grad-CAM"""
        if self.model is None:
            print("❌ No model loaded")
            return None
        
        # Preprocess image
        input_tensor, original_image = self.preprocess_image(image_path)
        if input_tensor is None:
            return None
        
        # Get target layer
        target_layer = self.get_target_layer()
        
        # Create Grad-CAM
        grad_cam = GradCAM(self.model, target_layer)
        
        # Generate CAM
        cam, predicted_class = grad_cam.generate_cam(input_tensor)
        
        if cam is None:
            print("❌ Could not generate Grad-CAM")
            grad_cam.remove_hooks()
            return None
        
        # Get prediction details
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence = probabilities.max().item()
            predicted_label = "FAKE" if predicted_class == 1 else "REAL"
        
        # Resize CAM to match original image
        cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Grad-CAM heatmap
        im1 = axes[0, 1].imshow(cam_resized, cmap='jet')
        axes[0, 1].set_title(f'Grad-CAM Heatmap\nPrediction: {predicted_label} ({confidence:.3f})')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Overlay
        overlay = original_image.astype(np.float32) / 255.0
        cam_colored = plt.cm.jet(cam_resized)[:, :, :3]
        overlay = 0.6 * overlay + 0.4 * cam_colored
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('Grad-CAM Overlay')
        axes[1, 0].axis('off')
        
        # Prediction probabilities
        probs = probabilities[0].cpu().numpy()
        labels = ['Real', 'Fake']
        colors = ['green', 'red']
        bars = axes[1, 1].bar(labels, probs, color=colors, alpha=0.7)
        axes[1, 1].set_title('Prediction Probabilities')
        axes[1, 1].set_ylabel('Probability')
        axes[1, 1].set_ylim(0, 1)
        
        # Add probability text on bars
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{prob:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📸 Grad-CAM visualization saved to {save_path}")
        
        plt.show()
        
        # Clean up
        grad_cam.remove_hooks()
        
        return {
            'cam': cam_resized,
            'prediction': predicted_label,
            'confidence': confidence,
            'probabilities': probs.tolist()
        }
    
    def analyze_layer_activations(self, image_path, save_dir=None):
        """Analyze and visualize activations at different layers"""
        if self.model is None:
            print("❌ No model loaded")
            return None
        
        input_tensor, original_image = self.preprocess_image(image_path)
        if input_tensor is None:
            return None
        
        # Hook to capture activations
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hooks for convolutional layers
        hooks = []
        conv_layers = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                hook = module.register_forward_hook(get_activation(name))
                hooks.append(hook)
                conv_layers.append(name)
        
        # Forward pass
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Visualize activations
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        for layer_name in conv_layers[:4]:  # Show first 4 conv layers
            if layer_name in activations:
                feature_maps = activations[layer_name][0]  # Remove batch dimension
                
                # Select first 16 feature maps
                n_features = min(16, feature_maps.shape[0])
                
                fig, axes = plt.subplots(4, 4, figsize=(12, 12))
                fig.suptitle(f'Feature Maps - {layer_name}', fontsize=16)
                
                for i in range(16):
                    row, col = i // 4, i % 4
                    if i < n_features:
                        feature_map = feature_maps[i].cpu().numpy()
                        axes[row, col].imshow(feature_map, cmap='viridis')
                        axes[row, col].set_title(f'Filter {i}')
                    axes[row, col].axis('off')
                
                plt.tight_layout()
                
                if save_dir:
                    save_path = os.path.join(save_dir, f'activations_{layer_name.replace(".", "_")}.png')
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
                plt.show()
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return activations
    
    def compare_model_focus(self, image_paths, model_paths, save_path=None):
        """Compare what different models focus on"""
        if not isinstance(image_paths, list):
            image_paths = [image_paths]
        
        if not isinstance(model_paths, list):
            model_paths = [model_paths]
        
        fig, axes = plt.subplots(len(image_paths), len(model_paths) + 1, 
                                figsize=(4 * (len(model_paths) + 1), 4 * len(image_paths)))
        
        if len(image_paths) == 1:
            axes = axes.reshape(1, -1)
        
        for img_idx, image_path in enumerate(image_paths):
            # Load original image
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # Show original
            axes[img_idx, 0].imshow(original_image)
            axes[img_idx, 0].set_title(f'Original\n{os.path.basename(image_path)}')
            axes[img_idx, 0].axis('off')
            
            # Generate Grad-CAM for each model
            for model_idx, model_path in enumerate(model_paths):
                # Determine model type
                model_type = 'vit' if 'vit' in model_path.lower() else 'cnn'
                
                # Create visualizer
                visualizer = InterpretabilityVisualizer(model_path, model_type)
                
                if visualizer.model is not None:
                    # Generate Grad-CAM
                    input_tensor, _ = visualizer.preprocess_image(image_path)
                    target_layer = visualizer.get_target_layer()
                    grad_cam = GradCAM(visualizer.model, target_layer)
                    cam, predicted_class = grad_cam.generate_cam(input_tensor)
                    
                    if cam is not None:
                        # Resize and overlay
                        cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
                        overlay = original_image.astype(np.float32) / 255.0
                        cam_colored = plt.cm.jet(cam_resized)[:, :, :3]
                        overlay = 0.6 * overlay + 0.4 * cam_colored
                        
                        # Get prediction
                        with torch.no_grad():
                            output = visualizer.model(input_tensor)
                            probabilities = F.softmax(output, dim=1)
                            confidence = probabilities.max().item()
                            predicted_label = "FAKE" if predicted_class == 1 else "REAL"
                        
                        axes[img_idx, model_idx + 1].imshow(overlay)
                        model_name = os.path.basename(model_path).replace('.pth', '')
                        axes[img_idx, model_idx + 1].set_title(f'{model_name}\n{predicted_label} ({confidence:.3f})')
                    
                    grad_cam.remove_hooks()
                
                axes[img_idx, model_idx + 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📸 Model comparison saved to {save_path}")
        
        plt.show()

# Batch analysis utilities
class BatchInterpretability:
    def __init__(self, model_paths):
        self.visualizers = {}
        for model_path in model_paths:
            model_type = 'vit' if 'vit' in model_path.lower() else 'cnn'
            model_name = os.path.basename(model_path).replace('.pth', '')
            self.visualizers[model_name] = InterpretabilityVisualizer(model_path, model_type)
    
    def analyze_dataset_samples(self, data_dir, num_samples=10, save_dir=None):
        """Analyze random samples from dataset"""
        # Get sample images
        real_dir = os.path.join(data_dir, 'test/real')
        fake_dir = os.path.join(data_dir, 'test/fake')
        
        real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir)[:num_samples//2]]
        fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)[:num_samples//2]]
        
        all_images = real_images + fake_images
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        # Analyze each image with each model
        results = {}
        
        for img_path in all_images:
            img_name = os.path.basename(img_path)
            results[img_name] = {}
            
            for model_name, visualizer in self.visualizers.items():
                if visualizer.model is not None:
                    save_path = os.path.join(save_dir, f'{model_name}_{img_name}.png') if save_dir else None
                    result = visualizer.generate_gradcam(img_path, save_path)
                    results[img_name][model_name] = result
        
        return results
    
    def generate_summary_report(self, results, save_path=None):
        """Generate summary report of interpretability analysis"""
        report = {
            'total_images': len(results),
            'models_analyzed': list(next(iter(results.values())).keys()),
            'analysis_summary': {}
        }
        
        # Analyze predictions by model
        for model_name in report['models_analyzed']:
            model_results = []
            for img_results in results.values():
                if model_name in img_results and img_results[model_name]:
                    model_results.append(img_results[model_name])
            
            if model_results:
                predictions = [r['prediction'] for r in model_results]
                confidences = [r['confidence'] for r in model_results]
                
                report['analysis_summary'][model_name] = {
                    'total_predictions': len(predictions),
                    'fake_predictions': predictions.count('FAKE'),
                    'real_predictions': predictions.count('REAL'),
                    'average_confidence': np.mean(confidences),
                    'confidence_std': np.std(confidences)
                }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📋 Summary report saved to {save_path}")
        
        return report

def main():
    """Main function for interpretability analysis"""
    parser = argparse.ArgumentParser(description='Model Interpretability Analysis')
    parser.add_argument('--mode', type=str, choices=['single', 'compare', 'batch'], 
                       default='single', help='Analysis mode')
    parser.add_argument('--image', type=str, help='Path to image for single analysis')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--models', nargs='+', help='Paths to multiple models for comparison')
    parser.add_argument('--data_dir', type=str, default='data/celeb_df/splits', 
                       help='Path to dataset for batch analysis')
    parser.add_argument('--save_dir', type=str, default='results/interpretability', 
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    print("🔍 Model Interpretability Analysis")
    print("=" * 50)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.mode == 'single':
        if not args.image or not args.model:
            print("❌ Please provide --image and --model for single analysis")
            return
        
        print(f"🔍 Analyzing single image: {args.image}")
        model_type = 'vit' if 'vit' in args.model.lower() else 'cnn'
        visualizer = InterpretabilityVisualizer(args.model, model_type)
        
        save_path = os.path.join(args.save_dir, f'gradcam_{os.path.basename(args.image)}.png')
        result = visualizer.generate_gradcam(args.image, save_path)
        
        if result:
            print(f"✅ Analysis complete. Prediction: {result['prediction']} ({result['confidence']:.3f})")
    
    elif args.mode == 'compare':
        if not args.image or not args.models:
            print("❌ Please provide --image and --models for comparison")
            return
        
        print(f"🔍 Comparing models on image: {args.image}")
        save_path = os.path.join(args.save_dir, f'model_comparison_{os.path.basename(args.image)}.png')
        
        # Create a dummy visualizer for the comparison method
        visualizer = InterpretabilityVisualizer(args.models[0])
        visualizer.compare_model_focus(args.image, args.models, save_path)
        
        print("✅ Model comparison complete")
    
    elif args.mode == 'batch':
        if not args.models:
            print("❌ Please provide --models for batch analysis")
            return
        
        print(f"🔍 Running batch analysis on dataset: {args.data_dir}")
        batch_analyzer = BatchInterpretability(args.models)
        
        results = batch_analyzer.analyze_dataset_samples(
            args.data_dir, 
            num_samples=20, 
            save_dir=args.save_dir
        )
        
        # Generate summary
        report_path = os.path.join(args.save_dir, 'interpretability_report.json')
        report = batch_analyzer.generate_summary_report(results, report_path)
        
        print("✅ Batch analysis complete")
        print(f"📊 Analyzed {report['total_images']} images with {len(report['models_analyzed'])} models")
    
    print("\n✅ Interpretability analysis finished!")

if __name__ == "__main__":
    main()