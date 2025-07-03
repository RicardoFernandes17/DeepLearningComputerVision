#!/usr/bin/env python3
"""
01_setup_and_run.py
Fixed Project Launcher and Setup Script for Deepfake Detection
This script helps you set up and run the complete deepfake detection system
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner():
    print("="*70)
    print("🕵️  ENHANCED DEEPFAKE DETECTION PROJECT - MAIN LAUNCHER")
    print("="*70)
    print("📚 Deep Learning for Computer Vision Assessment")
    print("🎯 CNN vs ViT comparison with real-time detection")
    print("🔍 Advanced interpretability with Grad-CAM")
    print("="*70)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7+ is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    # Enhanced requirements list
    requirements = [
        "torch>=1.9.0",
        "torchvision>=0.10.0", 
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.24.0",
        "Pillow>=8.0.0",
        "tqdm>=4.60.0",
        "albumentations>=1.1.0",
        "streamlit>=1.10.0",
        "plotly>=5.0.0",
        "pandas>=1.3.0",
        "kaggle>=1.5.12",
        "timm>=0.6.0",  # NEW: Vision Transformers
        "scipy>=1.7.0"  # NEW: Statistical analysis
    ]
    
    for package in requirements:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"⚠️  Failed to install {package}, trying without version constraint...")
            package_name = package.split(">=")[0]
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except:
                print(f"❌ Could not install {package_name}")
    
    print("✅ Package installation completed!")

def create_project_structure():
    """Create necessary directories"""
    print("\n📁 Creating enhanced project structure...")
    
    directories = [
        "data/celeb_df/raw",
        "data/celeb_df/processed/real",
        "data/celeb_df/processed/fake",
        "data/celeb_df/splits/train/real",
        "data/celeb_df/splits/train/fake",
        "data/celeb_df/splits/val/real",
        "data/celeb_df/splits/val/fake",
        "data/celeb_df/splits/test/real",
        "data/celeb_df/splits/test/fake",
        "checkpoints",
        "results",
        "results/comparison",           # NEW: Model comparison results
        "results/interpretability",    # NEW: Grad-CAM and interpretability
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Enhanced project structure created!")

def check_kaggle_setup():
    """Check if Kaggle API is set up"""
    print("\n🔑 Checking Kaggle API setup...")
    
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if kaggle_json.exists():
        print("✅ Kaggle credentials found!")
        return True
    else:
        print("❌ Kaggle credentials not found!")
        print("\n📋 To set up Kaggle API:")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Scroll down to 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. Move kaggle.json to ~/.kaggle/kaggle.json")
        print("\nCommands:")
        print("mkdir -p ~/.kaggle")
        print("mv ~/Downloads/kaggle.json ~/.kaggle/")
        print("chmod 600 ~/.kaggle/kaggle.json")
        return False

def show_enhanced_menu():
    """Show the enhanced main usage menu"""
    print("\n🎮 ENHANCED DEEPFAKE DETECTION SYSTEM - MAIN MENU")
    print("=" * 55)
    print("📊 DATA & SETUP")
    print("1. 📥 Download and setup dataset")
    
    print("\n🧠 MODEL TRAINING")
    print("2. 🏋️  Train CNN model (Simple - Recommended)")
    print("3. 🚀 Train CNN model (Advanced)")
    print("4. 🤖 Train Vision Transformer (ViT)")
    
    print("\n🔍 MODEL COMPARISON & ANALYSIS")
    print("5. ⚖️  Compare CNN vs ViT models")
    print("6. 🔬 Generate Grad-CAM interpretability")
    print("7. 📊 Comprehensive model analysis")
    
    print("\n🎥 REAL-TIME DETECTION")
    print("8. 🎥 Real-time webcam detection")
    print("9. 🧪 Test single image")
    
    print("\n🌐 INTERFACES & RESULTS")
    print("10. 🌐 Launch web interface")
    print("11. 📊 View training results")
    print("12. 🔧 Fix model file paths")
    
    print("\n❓ HELP & INFO")
    print("13. ❓ Help and documentation")
    print("14. 🚪 Exit")
    print("=" * 55)

def run_script_safely(script_name, description, args=None):
    """Safely run a Python script with error handling"""
    print(f"\n{description}...")
    
    if not os.path.exists(script_name):
        print(f"❌ Script not found: {script_name}")
        return False
    
    try:
        cmd = [sys.executable, script_name]
        if args:
            cmd.extend(args)
        
        print(f"🚀 Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            return True
        else:
            print(f"❌ {description} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False

def download_dataset():
    """Download and setup the dataset"""
    if not check_kaggle_setup():
        print("⚠️  Please set up Kaggle API first!")
        return False
    
    return run_script_safely("02_download_celeb_df.py", "Dataset download")

def train_cnn_simple():
    """Train CNN using simple method"""
    if not os.path.exists("data/celeb_df/splits"):
        print("❌ Dataset not found! Please download the dataset first.")
        return False
    
    return run_script_safely("03_simple_training.py", "Simple CNN training")

def train_cnn_advanced():
    """Train CNN using advanced method"""
    if not os.path.exists("data/celeb_df/splits"):
        print("❌ Dataset not found! Please download the dataset first.")
        return False
    
    return run_script_safely("04_advanced_training.py", "Advanced CNN training")

def train_vit():
    """Train Vision Transformer model"""
    if not os.path.exists("data/celeb_df/splits"):
        print("❌ Dataset not found! Please download the dataset first.")
        return False
    
    # Check if timm is available
    try:
        import timm
        print("✅ timm library available for ViT training")
    except ImportError:
        print("⚠️  Installing timm library for Vision Transformers...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "timm"])
            print("✅ timm installed successfully")
        except:
            print("❌ Failed to install timm. Please install manually: pip install timm")
            return False
    
    return run_script_safely("09_vit_model.py", "Vision Transformer training")

def compare_models():
    """Run comprehensive CNN vs ViT comparison"""
    print("\n⚖️  Starting CNN vs ViT model comparison...")
    
    # Check for available models
    model_files = {
        'enhanced_deepfake_detector.pth': 'cnn',
        'simple_deepfake_detector.pth': 'cnn',
        'vit_deepfake_detector.pth': 'vit',
        'enhanced_vit_deepfake_detector.pth': 'vit',
        'best_deepfake_detector.pth': 'cnn'
    }
    
    available_models = []
    model_paths = []
    model_types = []
    
    for path, model_type in model_files.items():
        if os.path.exists(path):
            available_models.append(path)
            model_paths.append(path)
            model_types.append(model_type)
    
    if len(available_models) < 2:
        print("❌ Need at least 2 trained models for comparison!")
        print(f"Available models: {available_models}")
        print("Please train more models first.")
        return False
    
    print(f"📊 Found {len(available_models)} models for comparison")
    for i, (model, mtype) in enumerate(zip(available_models, model_types)):
        print(f"  {i+1}. {model} ({mtype.upper()})")
    
    # Create a simple comparison script call
    args = ["--models"] + model_paths[:4]  # Limit to first 4 models
    args.extend(["--model_types"] + model_types[:4])
    
    return run_script_safely("11_model_comparison.py", "Model comparison", args)

def generate_interpretability():
    """Generate Grad-CAM and interpretability analysis"""
    print("\n🔬 Starting interpretability analysis...")
    
    # Check for available models
    model_files = [
        'enhanced_deepfake_detector.pth',
        'simple_deepfake_detector.pth',
        'vit_deepfake_detector.pth'
    ]
    
    available_models = [m for m in model_files if os.path.exists(m)]
    
    if not available_models:
        print("❌ No trained models found!")
        print("Please train a model first.")
        return False
    
    print(f"🔍 Found models: {', '.join(available_models)}")
    
    # Ask user for analysis type
    print("\nSelect analysis type:")
    print("1. Single image analysis")
    print("2. Batch analysis on test dataset")
    print("3. Model comparison analysis")
    
    try:
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            image_path = input("Enter image path: ").strip()
            if not os.path.exists(image_path):
                print("❌ Image not found!")
                return False
            
            model_path = available_models[0]  # Use first available model
            args = ["--mode", "single", "--image", image_path, "--model", model_path]
        
        elif choice == "2":
            args = ["--mode", "batch", "--models"] + available_models[:3]  # Limit to 3 models
        
        elif choice == "3":
            if len(available_models) < 2:
                print("❌ Need at least 2 models for comparison!")
                return False
            
            image_path = input("Enter image path for comparison: ").strip()
            if not os.path.exists(image_path):
                print("❌ Image not found!")
                return False
            
            args = ["--mode", "compare", "--image", image_path, "--models"] + available_models[:3]
        
        else:
            print("❌ Invalid choice!")
            return False
        
        return run_script_safely("10_interpretability.py", "Interpretability analysis", args)
        
    except KeyboardInterrupt:
        print("\n❌ Analysis cancelled by user")
        return False
    except Exception as e:
        print(f"❌ Error in interpretability analysis: {e}")
        return False

def comprehensive_analysis():
    """Run comprehensive model analysis including training metrics"""
    print("\n📊 Starting comprehensive model analysis...")
    
    # Check for training results
    result_files = [
        "results/simple_training_results.json",
        "results/training_history.json", 
        "results/vit_training_results.json"
    ]
    
    available_results = [f for f in result_files if os.path.exists(f)]
    
    if not available_results:
        print("❌ No training results found!")
        print("Please train some models first.")
        return False
    
    print(f"📈 Found training results: {len(available_results)} files")
    
    # Check if we can run model comparison
    model_files = [
        'enhanced_deepfake_detector.pth',
        'simple_deepfake_detector.pth', 
        'vit_deepfake_detector.pth'
    ]
    
    available_models = [m for m in model_files if os.path.exists(m)]
    
    if len(available_models) >= 2:
        print("🚀 Running model comparison as part of comprehensive analysis...")
        return compare_models()
    else:
        print("✅ Training analysis completed!")
        print("📊 Check results/ directory for training metrics")
        
        # Open results directory
        try:
            if sys.platform == "win32":
                os.startfile("results")
            elif sys.platform == "darwin":
                subprocess.run(["open", "results"])
            else:
                subprocess.run(["xdg-open", "results"])
        except:
            print(f"📁 Results location: {os.path.abspath('results')}")
        
        return True

def run_webcam_detection():
    """Run real-time webcam detection"""
    print("\n🎥 Starting real-time webcam detection...")
    
    # Check if model exists
    model_files = ['enhanced_deepfake_detector.pth', 'simple_deepfake_detector.pth', 
                   'best_deepfake_detector.pth', 'vit_deepfake_detector.pth']
    model_found = any(os.path.exists(f) for f in model_files)
    
    if not model_found:
        print("❌ No trained model found! Please train a model first.")
        return False
    
    print("🚀 Starting webcam detection...")
    print("Press 'q' in the video window to quit")
    
    # Try main detection script first, fallback to simple
    if os.path.exists("05_realtime_detection.py"):
        return run_script_safely("05_realtime_detection.py", "Real-time detection", ["--mode", "realtime"])
    elif os.path.exists("06_simple_realtime.py"):
        return run_script_safely("06_simple_realtime.py", "Simple real-time detection")
    else:
        print("❌ No detection script found!")
        return False

def test_single_image():
    """Test detection on a single image"""
    print("\n🧪 Single image testing...")
    
    # Check if model exists
    model_files = ['enhanced_deepfake_detector.pth', 'simple_deepfake_detector.pth', 
                   'best_deepfake_detector.pth', 'vit_deepfake_detector.pth']
    model_found = any(os.path.exists(f) for f in model_files)
    
    if not model_found:
        print("❌ No trained model found! Please train a model first.")
        return False
    
    try:
        image_path = input("📸 Enter image path: ").strip()
        
        if not os.path.exists(image_path):
            print("❌ Image not found!")
            return False
        
        if os.path.exists("05_realtime_detection.py"):
            return run_script_safely("05_realtime_detection.py", "Single image test", 
                                    ["--mode", "test", "--image", image_path])
        else:
            print("❌ Detection script not found!")
            return False
    except KeyboardInterrupt:
        print("\n❌ Test cancelled by user")
        return False

def launch_web_interface():
    """Launch the Streamlit web interface"""
    print("\n🌐 Launching enhanced web interface...")
    
    if not os.path.exists("07_web_interface.py"):
        print("❌ Web interface script not found!")
        return False
    
    try:
        print("🚀 Starting Streamlit app...")
        print("🌐 The web interface will open in your browser")
        print("📱 Access it at: http://localhost:8501")
        print("Press Ctrl+C to stop the server")
        
        subprocess.run([sys.executable, "-m", "streamlit", "run", "07_web_interface.py"])
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 Web interface stopped")
        return True
    except Exception as e:
        print(f"❌ Error launching web interface: {e}")
        return False

def view_results():
    """View training results and plots"""
    print("\n📊 Viewing training results...")
    
    results_dir = "results"
    if not os.path.exists(results_dir):
        print("❌ No results found! Please train a model first.")
        return False
    
    # List available result files
    result_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith(('.json', '.png', '.jpg', '.csv', '.md')):
                result_files.append(os.path.join(root, file))
    
    if not result_files:
        print("❌ No result files found!")
        return False
    
    print("📁 Available result files:")
    for i, file in enumerate(result_files, 1):
        rel_path = os.path.relpath(file, results_dir)
        print(f"  {i}. {rel_path}")
    
    # Open results directory
    try:
        if sys.platform == "win32":
            os.startfile(results_dir)
        elif sys.platform == "darwin":
            subprocess.run(["open", results_dir])
        else:
            subprocess.run(["xdg-open", results_dir])
        
        print(f"✅ Results directory opened: {os.path.abspath(results_dir)}")
        return True
        
    except:
        print(f"📁 Results location: {os.path.abspath(results_dir)}")
        return True

def fix_model_paths():
    """Fix model file paths and compatibility issues"""
    print("\n🔧 Fixing model file paths...")
    
    # Simple model path fixing logic
    model_files = [
        'enhanced_deepfake_detector.pth',
        'simple_deepfake_detector.pth',
        'vit_deepfake_detector.pth',
        'best_deepfake_detector.pth'
    ]
    
    found_models = [f for f in model_files if os.path.exists(f)]
    
    if not found_models:
        print("❌ No model files found to fix!")
        return False
    
    print(f"🔍 Found {len(found_models)} model files:")
    for model in found_models:
        print(f"  ✅ {model}")
    
    # Create backup links for compatibility
    try:
        if os.path.exists('enhanced_deepfake_detector.pth') and not os.path.exists('best_deepfake_detector.pth'):
            import shutil
            shutil.copy2('enhanced_deepfake_detector.pth', 'best_deepfake_detector.pth')
            print("🔗 Created compatibility link: best_deepfake_detector.pth")
        
        print("✅ Model paths verified and fixed!")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing model paths: {e}")
        return False

def show_enhanced_help():
    """Show enhanced help and documentation"""
    print("\n❓ ENHANCED HELP AND DOCUMENTATION")
    print("=" * 60)
    
    print("""
📖 QUICK START GUIDE:

1. 🔧 SETUP:
   - Run option 1 to download dataset (requires Kaggle API)
   - Install required packages when prompted

2. 🧠 TRAINING (Choose your path):
   - Option 2: Simple CNN (5-10 min, 85-90% accuracy)
   - Option 3: Advanced CNN (15-20 min, 90-95% accuracy)  
   - Option 4: Vision Transformer (20-30 min, 90-95% accuracy)

3. 🔍 ANALYSIS & COMPARISON:
   - Option 5: Compare CNN vs ViT performance
   - Option 6: Generate Grad-CAM visualizations
   - Option 7: Comprehensive analysis with statistical tests

4. 🎥 TESTING:
   - Option 8: Real-time webcam detection
   - Option 9: Single image testing

5. 🌐 DEMO:
   - Option 10: Professional web interface

📋 TROUBLESHOOTING:

❌ "Script not found" errors:
   - Make sure all Python files are in the same directory
   - Check that file names match exactly

❌ Import errors:
   - Install missing packages: pip install package_name
   - Use option 1 to reinstall all requirements

❌ Model training fails:
   - Check if dataset was downloaded (option 1)
   - Ensure sufficient disk space (>2GB)
   - Try simple training before advanced

❌ GPU out of memory:
   - Reduce batch size in training scripts
   - Close other applications
   - Use CPU-only training

🎯 EXPECTED PERFORMANCE:
- CNN Training Time: 5-20 minutes
- ViT Training Time: 15-30 minutes  
- Accuracy: 85-95% depending on architecture
- Real-time FPS: 15-30 depending on hardware

📁 KEY FILES:
- 02_download_celeb_df.py: Dataset downloader
- 03_simple_training.py: Fast CNN training
- 04_advanced_training.py: Advanced CNN training
- 09_vit_model.py: Vision Transformer training
- 10_interpretability.py: Grad-CAM analysis
- 11_model_comparison.py: CNN vs ViT comparison
- 07_web_interface.py: Streamlit web app

💡 TIPS:
- Train at least CNN and ViT for comparison
- Use GPU for faster training if available
- Generate interpretability analysis for insights
- Check results/ folder for outputs
    """)

def main():
    """Enhanced main menu loop with better error handling"""
    print_banner()
    
    # Initial setup check
    if not check_python_version():
        return
    
    # Ask if user wants to install requirements
    try:
        install_deps = input("\n🔧 Install/update required packages? (y/n): ").lower().strip()
        if install_deps in ['y', 'yes']:
            install_requirements()
    except KeyboardInterrupt:
        print("\n❌ Setup cancelled by user")
        return
    
    # Create project structure
    create_project_structure()
    
    # Main menu loop
    while True:
        try:
            show_enhanced_menu()
            choice = input("\n🎯 Enter your choice (1-14): ").strip()
            
            if choice == '1':
                download_dataset()
                
            elif choice == '2':
                train_cnn_simple()
                
            elif choice == '3':
                train_cnn_advanced()
                
            elif choice == '4':
                train_vit()
                
            elif choice == '5':
                compare_models()
                
            elif choice == '6':
                generate_interpretability()
                
            elif choice == '7':
                comprehensive_analysis()
                
            elif choice == '8':
                run_webcam_detection()
                
            elif choice == '9':
                test_single_image()
                
            elif choice == '10':
                launch_web_interface()
                
            elif choice == '11':
                view_results()
                
            elif choice == '12':
                fix_model_paths()
                
            elif choice == '13':
                show_enhanced_help()
                
            elif choice == '14':
                print("\n👋 Thanks for using the Enhanced Deepfake Detection System!")
                print("🎓 Good luck with your assessment!")
                print("🏆 You now have CNN vs ViT comparison with interpretability!")
                break
                
            else:
                print("❌ Invalid choice! Please enter 1-14.")
            
            # Pause before showing menu again
            input("\n⏸️  Press Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again or check the help section (option 13)")
            input("⏸️  Press Enter to continue...")

if __name__ == "__main__":
    main()