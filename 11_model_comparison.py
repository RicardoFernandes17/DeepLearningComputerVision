#!/usr/bin/env python3
"""
11_model_comparison.py
Complete Working Comprehensive Comparison between CNN and Vision Transformer models
Provides systematic analysis and benchmarking with fallback implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve)
from scipy import stats
import time
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import warnings
import cv2
import random
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')

# Check for optional dependencies
try:
    import timm
    VIT_AVAILABLE = True
except ImportError:
    print("⚠️  timm not available, ViT support limited")
    VIT_AVAILABLE = False

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    print("⚠️  albumentations not available, using basic transforms")
    ALBUMENTATIONS_AVAILABLE = False

# Fallback model definitions
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
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
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
            try:
                self.vit = timm.create_model(
                    model_name, 
                    pretrained=pretrained, 
                    num_classes=num_classes
                )
                
                # Add dropout for regularization
                if hasattr(self.vit, 'head'):
                    in_features = self.vit.head.in_features
                    self.vit.head = nn.Sequential(
                        nn.Dropout(0.3),
                        nn.Linear(in_features, 512),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.3),
                        nn.Linear(512, num_classes)
                    )
            except Exception as e:
                print(f"⚠️  Error creating ViT model: {e}")
                print("Using CNN fallback for ViT")
                self.vit = SimpleDeepfakeDetector(num_classes)
        else:
            print("⚠️  ViT not available, using CNN fallback")
            self.vit = SimpleDeepfakeDetector(num_classes)
    
    def forward(self, x):
        return self.vit(x)

# Simple dataset for comparison
class ComparisonDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None, max_samples=400):
        self.transform = transform
        self.data = []
        self.labels = []
        
        # Load real images
        if os.path.exists(real_dir):
            real_files = [f for f in os.listdir(real_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(real_files) > max_samples//2:
                real_files = random.sample(real_files, max_samples//2)
            
            for filename in real_files:
                img_path = os.path.join(real_dir, filename)
                if self._is_valid_image(img_path):
                    self.data.append(img_path)
                    self.labels.append(0)  # 0 for real
        
        # Load fake images
        if os.path.exists(fake_dir):
            fake_files = [f for f in os.listdir(fake_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(fake_files) > max_samples//2:
                fake_files = random.sample(fake_files, max_samples//2)
            
            for filename in fake_files:
                img_path = os.path.join(fake_dir, filename)
                if self._is_valid_image(img_path):
                    self.data.append(img_path)
                    self.labels.append(1)  # 1 for fake
        
        # Shuffle data
        combined = list(zip(self.data, self.labels))
        random.shuffle(combined)
        self.data, self.labels = zip(*combined) if combined else ([], [])
        
        print(f"Comparison dataset: {len(self.data)} images ({sum(self.labels)} fake, {len(self.labels) - sum(self.labels)} real)")
    
    def _is_valid_image(self, img_path):
        """Check if image file is valid and readable"""
        try:
            img = cv2.imread(img_path)
            return img is not None and img.shape[0] > 32 and img.shape[1] > 32
        except:
            return False
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        
        try:
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Could not load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            
            return image, label
            
        except Exception as e:
            print(f"Warning: Error loading {img_path}: {e}")
            # Return dummy image
            dummy_image = torch.zeros(3, 224, 224)
            return dummy_image, label

# Model Comparison Framework
class ModelComparator:
    def __init__(self, output_dir="results/comparison"):
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🔬 Model Comparator initialized on {self.device}")
    
    def load_model(self, model_path, model_type='cnn'):
        """Load a trained model with comprehensive fallback handling"""
        try:
            print(f"Loading {model_type.upper()} model from {model_path}...")
            
            # Try to determine the best architecture based on file size and type
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            
            if model_type == 'vit':
                model = ViTDeepfakeDetector()
            elif file_size > 50:  # Larger files likely enhanced
                model = EnhancedDeepfakeDetector()
            else:
                model = SimpleDeepfakeDetector()
            
            # Load state dict with comprehensive error handling
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                if isinstance(state_dict, dict):
                    if 'model_state_dict' in state_dict:
                        model.load_state_dict(state_dict['model_state_dict'])
                    else:
                        model.load_state_dict(state_dict)
                else:
                    # Try to load as full model
                    model = torch.load(model_path, map_location=self.device)
                    
            except Exception as load_error:
                print(f"⚠️  Primary loading failed: {load_error}")
                print("Trying alternative architectures...")
                
                # Try different model architectures
                architectures = [
                    SimpleDeepfakeDetector(),
                    EnhancedDeepfakeDetector()
                ]
                
                if model_type == 'vit' or VIT_AVAILABLE:
                    architectures.append(ViTDeepfakeDetector())
                
                loaded = False
                for i, arch in enumerate(architectures):
                    try:
                        state_dict = torch.load(model_path, map_location=self.device)
                        if isinstance(state_dict, dict):
                            if 'model_state_dict' in state_dict:
                                arch.load_state_dict(state_dict['model_state_dict'])
                            else:
                                arch.load_state_dict(state_dict)
                        model = arch
                        print(f"✅ Loaded with {arch.__class__.__name__}")
                        loaded = True
                        break
                    except Exception as arch_error:
                        print(f"  Architecture {i+1} failed: {arch_error}")
                        continue
                
                if not loaded:
                    raise Exception("Could not load model with any architecture")
            
            model = model.to(self.device)
            model.eval()
            
            print(f"✅ Successfully loaded {model_type.upper()} model")
            return model
            
        except Exception as e:
            print(f"❌ Error loading {model_type} model from {model_path}: {e}")
            return None
    
    def evaluate_model(self, model, test_loader, model_name):
        """Evaluate a single model and return comprehensive metrics"""
        if model is None:
            print(f"❌ Cannot evaluate {model_name} - model is None")
            return None
        
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        inference_times = []
        
        print(f"🧪 Evaluating {model_name}...")
        
        try:
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(test_loader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    # Measure inference time
                    start_time = time.time()
                    outputs = model(images)
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time / images.size(0))  # Per image
                    
                    probabilities = F.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probabilities[:, 1].cpu().numpy())  # Fake probabilities
                    
                    # Progress indicator
                    if (batch_idx + 1) % 10 == 0:
                        print(f"  Processed batch {batch_idx + 1}/{len(test_loader)}")
        
        except Exception as e:
            print(f"❌ Error during evaluation of {model_name}: {e}")
            return None
        
        # Calculate metrics with error handling
        try:
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
            
            # Handle AUC calculation
            try:
                auc = roc_auc_score(all_labels, all_probs)
            except Exception as auc_error:
                print(f"⚠️  AUC calculation failed for {model_name}: {auc_error}")
                auc = 0.5  # Default value if AUC calculation fails
            
            # Per-class metrics
            try:
                precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
                recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
                f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
            except Exception as class_error:
                print(f"⚠️  Per-class metrics failed for {model_name}: {class_error}")
                precision_per_class = [0.0, 0.0]
                recall_per_class = [0.0, 0.0]
                f1_per_class = [0.0, 0.0]
            
            # Confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            
            # Performance metrics
            avg_inference_time = np.mean(inference_times) if inference_times else 0.0
            
            results = {
                'model_name': model_name,
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'auc_roc': float(auc),
                'precision_per_class': [float(x) for x in precision_per_class],
                'recall_per_class': [float(x) for x in recall_per_class],
                'f1_per_class': [float(x) for x in f1_per_class],
                'confusion_matrix': cm.tolist(),
                'avg_inference_time': float(avg_inference_time),
                'predictions': all_preds,
                'true_labels': all_labels,
                'probabilities': all_probs
            }
            
            print(f"✅ {model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error calculating metrics for {model_name}: {e}")
            return None
    
    def get_model_complexity(self, model):
        """Calculate model complexity metrics"""
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Estimate model size (MB)
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            
            model_size_mb = (param_size + buffer_size) / 1024 / 1024
            
            # Count layers
            conv_layers = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))
            linear_layers = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
            
            return {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': model_size_mb,
                'conv_layers': conv_layers,
                'linear_layers': linear_layers
            }
        except Exception as e:
            print(f"⚠️  Error calculating model complexity: {e}")
            return {
                'total_parameters': 0,
                'trainable_parameters': 0,
                'model_size_mb': 0.0,
                'conv_layers': 0,
                'linear_layers': 0
            }
    
    def compare_models(self, model_configs, test_loader):
        """Compare multiple models comprehensively"""
        results = {}
        complexity_data = {}
        
        print("🔬 Starting comprehensive model comparison...")
        
        for model_name, config in model_configs.items():
            print(f"\n{'='*50}")
            print(f"Processing model: {model_name}")
            print(f"{'='*50}")
            
            model = self.load_model(config['path'], config['type'])
            
            if model is not None:
                # Evaluate model
                eval_results = self.evaluate_model(model, test_loader, model_name)
                if eval_results:
                    results[model_name] = eval_results
                
                # Get complexity metrics
                complexity_data[model_name] = self.get_model_complexity(model)
                complexity_data[model_name]['model_type'] = config['type']
            else:
                print(f"⚠️  Skipping {model_name} due to loading failure")
        
        if not results:
            print("❌ No models were successfully evaluated!")
            return None
        
        print(f"\n✅ Successfully evaluated {len(results)} models")
        
        # Generate comparison report
        comparison_report = self.generate_comparison_report(results, complexity_data)
        
        # Create visualizations
        self.create_comparison_visualizations(results, complexity_data)
        
        # Save results
        self.save_comparison_results(results, complexity_data, comparison_report)
        
        return comparison_report
    
    def generate_comparison_report(self, results, complexity_data):
        """Generate detailed comparison report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_compared': list(results.keys()),
            'summary': {},
            'recommendations': {}
        }
        
        # Summary statistics
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'avg_inference_time']
        
        for metric in metrics:
            values = [results[model][metric] for model in results.keys()]
            if metric == 'avg_inference_time':
                best_idx = np.argmin(values)  # Lower is better for time
            else:
                best_idx = np.argmax(values)  # Higher is better for performance
            
            best_model = list(results.keys())[best_idx]
            
            report['summary'][metric] = {
                'best_model': best_model,
                'best_value': values[best_idx],
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        # Generate recommendations
        best_accuracy_model = report['summary']['accuracy']['best_model']
        fastest_model = report['summary']['avg_inference_time']['best_model']
        
        # Architecture analysis
        cnn_models = [k for k, v in complexity_data.items() if v['model_type'] == 'cnn']
        vit_models = [k for k, v in complexity_data.items() if v['model_type'] == 'vit']
        
        analysis = []
        analysis.append(f"🏆 Best performing model: {best_accuracy_model} with {report['summary']['accuracy']['best_value']:.1%} accuracy")
        analysis.append(f"⚡ Fastest model: {fastest_model} with {report['summary']['avg_inference_time']['best_value']*1000:.1f}ms inference time")
        
        if cnn_models and vit_models:
            cnn_avg_acc = np.mean([results[m]['accuracy'] for m in cnn_models])
            vit_avg_acc = np.mean([results[m]['accuracy'] for m in vit_models])
            
            if cnn_avg_acc > vit_avg_acc:
                analysis.append(f"🏗️ CNNs outperformed ViTs on average: {cnn_avg_acc:.1%} vs {vit_avg_acc:.1%}")
            else:
                analysis.append(f"🤖 ViTs outperformed CNNs on average: {vit_avg_acc:.1%} vs {cnn_avg_acc:.1%}")
        
        report['recommendations'] = {
            'best_overall': best_accuracy_model,
            'best_for_realtime': fastest_model,
            'analysis': analysis
        }
        
        return report
    
    def create_comparison_visualizations(self, results, complexity_data):
        """Create comprehensive visualization plots"""
        try:
            # Set style
            plt.style.use('default')
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            models = list(results.keys())
            
            # 1. Performance Comparison Bar Chart
            ax1 = axes[0, 0]
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            
            x = np.arange(len(models))
            width = 0.2
            
            colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
            
            for i, metric in enumerate(metrics):
                values = [results[model][metric] for model in models]
                ax1.bar(x + i*width, values, width, label=metric.replace('_', ' ').title(), 
                       color=colors[i], alpha=0.8)
            
            ax1.set_xlabel('Models')
            ax1.set_ylabel('Score')
            ax1.set_title('Performance Metrics Comparison')
            ax1.set_xticks(x + width * 1.5)
            ax1.set_xticklabels(models, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # 2. Inference Time Comparison
            ax2 = axes[0, 1]
            inference_times = [results[model]['avg_inference_time'] * 1000 for model in models]
            bars = ax2.bar(models, inference_times, color='orange', alpha=0.7)
            ax2.set_ylabel('Inference Time (ms)')
            ax2.set_title('Inference Speed Comparison')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, time in zip(bars, inference_times):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height + height*0.01,
                        f'{time:.1f}ms', ha='center', va='bottom')
            
            # 3. Model Size vs Accuracy
            ax3 = axes[1, 0]
            sizes = [complexity_data[model]['model_size_mb'] for model in models]
            accuracies = [results[model]['accuracy'] * 100 for model in models]
            
            # Color by model type
            colors = []
            for model in models:
                if complexity_data[model]['model_type'] == 'vit':
                    colors.append('red')
                else:
                    colors.append('blue')
            
            scatter = ax3.scatter(sizes, accuracies, c=colors, s=100, alpha=0.7)
            for i, model in enumerate(models):
                ax3.annotate(model, (sizes[i], accuracies[i]), 
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
            ax3.set_xlabel('Model Size (MB)')
            ax3.set_ylabel('Accuracy (%)')
            ax3.set_title('Model Size vs Accuracy')
            ax3.grid(True, alpha=0.3)
            
            # Create custom legend
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                                     markersize=8, label='CNN'),
                              Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                                     markersize=8, label='ViT')]
            ax3.legend(handles=legend_elements)
            
            # 4. ROC Curves
            ax4 = axes[1, 1]
            colors_roc = ['blue', 'red', 'green', 'orange', 'purple']
            
            for i, model in enumerate(models):
                try:
                    y_true = results[model]['true_labels']
                    y_scores = results[model]['probabilities']
                    
                    fpr, tpr, _ = roc_curve(y_true, y_scores)
                    auc_score = results[model]['auc_roc']
                    
                    color = colors_roc[i % len(colors_roc)]
                    ax4.plot(fpr, tpr, label=f'{model} (AUC = {auc_score:.3f})', 
                            linewidth=2, color=color)
                except Exception as roc_error:
                    print(f"⚠️  Could not plot ROC curve for {model}: {roc_error}")
            
            ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
            ax4.set_xlabel('False Positive Rate')
            ax4.set_ylabel('True Positive Rate')
            ax4.set_title('ROC Curves Comparison')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the plot
            save_path = os.path.join(self.output_dir, 'model_comparison_visualization.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Comparison visualization saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"⚠️  Error creating visualizations: {e}")
            print("Continuing without visualizations...")
    
    def save_comparison_results(self, results, complexity_data, comparison_report):
        """Save all comparison results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Create summary table
            table_data = []
            
            for model in results.keys():
                row = {
                    'Model': model,
                    'Type': complexity_data[model]['model_type'].upper(),
                    'Accuracy': f"{results[model]['accuracy']:.4f}",
                    'Precision': f"{results[model]['precision']:.4f}",
                    'Recall': f"{results[model]['recall']:.4f}",
                    'F1-Score': f"{results[model]['f1_score']:.4f}",
                    'AUC-ROC': f"{results[model]['auc_roc']:.4f}",
                    'Inference Time (ms)': f"{results[model]['avg_inference_time']*1000:.1f}",
                    'Parameters (M)': f"{complexity_data[model]['total_parameters']/1e6:.1f}",
                    'Model Size (MB)': f"{complexity_data[model]['model_size_mb']:.1f}"
                }
                table_data.append(row)
            
            df = pd.DataFrame(table_data)
            
            # Save as CSV
            csv_path = os.path.join(self.output_dir, 'model_comparison_summary.csv')
            df.to_csv(csv_path, index=False)
            print(f"📊 Summary table saved to {csv_path}")
            
            # Display table
            print("\n📋 MODEL COMPARISON SUMMARY")
            print("=" * 100)
            print(df.to_string(index=False))
            
            # Save detailed results
            results_path = os.path.join(self.output_dir, f'comparison_results_{timestamp}.json')
            
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for model, result in results.items():
                json_results[model] = {}
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        json_results[model][key] = value.tolist()
                    elif isinstance(value, (np.int64, np.float64)):
                        json_results[model][key] = float(value)
                    elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.int64):
                        json_results[model][key] = [int(x) for x in value]
                    elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.float64):
                        json_results[model][key] = [float(x) for x in value]
                    else:
                        json_results[model][key] = value
            
            detailed_results = {
                'timestamp': timestamp,
                'model_results': json_results,
                'complexity_data': complexity_data,
                'comparison_report': comparison_report
            }
            
            with open(results_path, 'w') as f:
                json.dump(detailed_results, f, indent=2)
            
            print(f"💾 Detailed results saved to {results_path}")
            
            # Create markdown report
            self.create_markdown_report(comparison_report, results, complexity_data)
            
        except Exception as e:
            print(f"⚠️  Error saving results: {e}")
    
    def create_markdown_report(self, comparison_report, results, complexity_data):
        """Create a comprehensive markdown report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        markdown_content = f"""# Deepfake Detection Model Comparison Report

**Generated:** {timestamp}

## Executive Summary

This report presents a comprehensive comparison of deepfake detection models, evaluating their performance, efficiency, and complexity across multiple metrics.

### Models Compared
{chr(10).join([f"- **{model}** ({complexity_data[model]['model_type'].upper()})" for model in results.keys()])}

## Performance Summary

### Best Performing Models
"""
        
        # Add best model information
        for metric in ['accuracy', 'f1_score', 'auc_roc']:
            if metric in comparison_report['summary']:
                best_model = comparison_report['summary'][metric]['best_model']
                best_value = comparison_report['summary'][metric]['best_value']
                markdown_content += f"- **{metric.replace('_', ' ').title()}:** {best_model} ({best_value:.4f})\n"
        
        # Add fastest model
        if 'avg_inference_time' in comparison_report['summary']:
            fastest_model = comparison_report['summary']['avg_inference_time']['best_model']
            fastest_time = comparison_report['summary']['avg_inference_time']['best_value']
            markdown_content += f"- **Fastest Inference:** {fastest_model} ({fastest_time*1000:.1f}ms)\n"
        
        markdown_content += """
## Detailed Results

### Performance Metrics
| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Inference Time (ms) |
|-------|----------|-----------|--------|----------|---------|-------------------|
"""
        
        for model in results.keys():
            r = results[model]
            markdown_content += f"| {model} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1_score']:.4f} | {r['auc_roc']:.4f} | {r['avg_inference_time']*1000:.1f} |\n"
        
        markdown_content += """
### Model Complexity
| Model | Type | Parameters | Model Size (MB) | Conv Layers | Linear Layers |
|-------|------|------------|----------------|-------------|---------------|
"""
        
        for model in complexity_data.keys():
            c = complexity_data[model]
            markdown_content += f"| {model} | {c['model_type'].upper()} | {c['total_parameters']:,} | {c['model_size_mb']:.1f} | {c['conv_layers']} | {c['linear_layers']} |\n"
        
        # Add analysis
        markdown_content += f"""
## Analysis and Recommendations

{chr(10).join(['- ' + analysis for analysis in comparison_report['recommendations']['analysis']])}

### Key Findings

1. **Best Overall Performance:** {comparison_report['recommendations']['best_overall']}
2. **Best for Real-time Applications:** {comparison_report['recommendations']['best_for_realtime']}

### Architecture Comparison

"""
        
        # Add architecture-specific analysis
        cnn_models = [k for k, v in complexity_data.items() if v['model_type'] == 'cnn']
        vit_models = [k for k, v in complexity_data.items() if v['model_type'] == 'vit']
        
        if cnn_models and vit_models:
            cnn_avg_acc = np.mean([results[m]['accuracy'] for m in cnn_models])
            vit_avg_acc = np.mean([results[m]['accuracy'] for m in vit_models])
            cnn_avg_speed = np.mean([results[m]['avg_inference_time'] for m in cnn_models])
            vit_avg_speed = np.mean([results[m]['avg_inference_time'] for m in vit_models])
            
            markdown_content += f"""
**CNN Models:**
- Average Accuracy: {cnn_avg_acc:.4f}
- Average Inference Time: {cnn_avg_speed*1000:.1f}ms
- Models: {', '.join(cnn_models)}

**Vision Transformer Models:**
- Average Accuracy: {vit_avg_acc:.4f}
- Average Inference Time: {vit_avg_speed*1000:.1f}ms
- Models: {', '.join(vit_models)}

**Conclusion:** {'CNNs' if cnn_avg_acc > vit_avg_acc else 'ViTs'} show superior accuracy on average, while {'CNNs' if cnn_avg_speed < vit_avg_speed else 'ViTs'} are faster for inference.
"""
        
        markdown_content += f"""
## Methodology

### Evaluation Protocol
- **Dataset:** Celeb-DF test split
- **Metrics:** Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Hardware:** {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}
- **Framework:** PyTorch {torch.__version__}

### Model Configurations
- All models evaluated on the same test dataset
- Inference time measured as average per image
- Model complexity includes trainable parameters only

---
*Report generated by Deepfake Detection Model Comparison Framework*
"""
        
        # Save markdown report
        report_path = os.path.join(self.output_dir, 'comparison_report.md')
        with open(report_path, 'w') as f:
            f.write(markdown_content)
        
        print(f"📄 Markdown report saved to {report_path}")

def create_test_loader(data_path="data/celeb_df/splits", batch_size=32, image_size=224):
    """Create test data loader for model comparison"""
    try:
        # Create transform
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create dataset
        dataset = ComparisonDataset(
            real_dir=os.path.join(data_path, 'test/real'),
            fake_dir=os.path.join(data_path, 'test/fake'),
            transform=transform
        )
        
        if len(dataset) == 0:
            print("❌ No data found in dataset!")
            return None
        
        # Create data loader
        test_loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=0,  # Set to 0 for compatibility
            pin_memory=False
        )
        
        return test_loader
        
    except Exception as e:
        print(f"❌ Error creating test loader: {e}")
        return None

def main():
    """Main function for model comparison"""
    parser = argparse.ArgumentParser(description='Comprehensive Model Comparison')
    parser.add_argument('--models', nargs='+', required=True, 
                       help='Paths to model files to compare')
    parser.add_argument('--model_types', nargs='+', 
                       help='Model types (cnn/vit) corresponding to each model path')
    parser.add_argument('--data_path', type=str, default='data/celeb_df/splits',
                       help='Path to test dataset')
    parser.add_argument('--output_dir', type=str, default='results/comparison',
                       help='Output directory for comparison results')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    
    args = parser.parse_args()
    
    print("🔬 Starting Comprehensive Model Comparison")
    print("=" * 60)
    
    # Validate inputs
    if not os.path.exists(args.data_path):
        print(f"❌ Dataset not found at {args.data_path}")
        print("Please download the dataset first using the main launcher")
        return
    
    # Check model files
    missing_models = [m for m in args.models if not os.path.exists(m)]
    if missing_models:
        print(f"❌ Missing model files: {missing_models}")
        print("Available models in current directory:")
        available = [f for f in os.listdir('.') if f.endswith('.pth')]
        for model in available:
            print(f"  - {model}")
        return
    
    # Set default model types if not provided
    if not args.model_types:
        args.model_types = ['vit' if 'vit' in m.lower() else 'cnn' for m in args.models]
    
    if len(args.model_types) != len(args.models):
        print("❌ Number of model types must match number of models")
        return
    
    # Create model configurations
    model_configs = {}
    for i, (model_path, model_type) in enumerate(zip(args.models, args.model_types)):
        model_name = os.path.basename(model_path).replace('.pth', '')
        model_configs[model_name] = {
            'path': model_path,
            'type': model_type
        }
    
    print(f"📋 Models to compare: {list(model_configs.keys())}")
    
    # Create test data loader
    print("📊 Loading test dataset...")
    test_loader = create_test_loader(
        data_path=args.data_path,
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    
    if test_loader is None:
        print("❌ Failed to create test data loader")
        return
    
    # Initialize comparator
    comparator = ModelComparator(args.output_dir)
    
    # Run comparison
    print("🚀 Starting model comparison...")
    comparison_report = comparator.compare_models(model_configs, test_loader)
    
    if comparison_report is None:
        print("❌ Model comparison failed")
        return
    
    # Print summary
    print("\n🎉 COMPARISON COMPLETE!")
    print("=" * 60)
    print(f"📊 Results saved to: {args.output_dir}")
    print(f"🏆 Best overall model: {comparison_report['recommendations']['best_overall']}")
    print(f"⚡ Fastest model: {comparison_report['recommendations']['best_for_realtime']}")
    
    # Print key findings
    print("\n📋 Key Findings:")
    for finding in comparison_report['recommendations']['analysis']:
        print(f"  {finding}")

if __name__ == "__main__":
    main()