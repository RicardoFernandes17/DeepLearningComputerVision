#!/usr/bin/env python3
"""
Setup and Run Script for Deepfake Detection Project
This script helps you set up and run the complete deepfake detection system
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_banner():
    print("="*70)
    print("🕵️  DEEPFAKE DETECTION PROJECT SETUP")
    print("="*70)
    print("📚 Deep Learning for Computer Vision Assessment")
    print("🎯 Real-time deepfake detection with OpenCV")
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
        "kaggle>=1.5.12"
    ]
    
    for package in requirements:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError:
            print(f"⚠️  Failed to install {package}, trying without version constraint...")
            package_name = package.split(">=")[0]
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
    
    print("✅ All packages installed successfully!")

def create_project_structure():
    """Create necessary directories"""
    print("\n📁 Creating project structure...")
    
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
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Project structure created!")

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

def show_usage_menu():
    """Show the main usage menu"""
    print("\n🎮 DEEPFAKE DETECTION SYSTEM")
    print("=" * 40)
    print("1. 📥 Download and setup dataset")
    print("2. 🏋️  Train the model")
    print("3. 🎥 Real-time webcam detection")
    print("4. 🌐 Launch web interface")
    print("5. 🧪 Test single image")
    print("6. 📊 View training results")
    print("7. ❓ Help and documentation")
    print("8. 🚪 Exit")
    print("=" * 40)

def download_dataset():
    """Download and setup the dataset"""
    print("\n📥 Starting dataset download...")
    
    if not check_kaggle_setup():
        print("⚠️  Please set up Kaggle API first!")
        return False
    
    try:
        # Run the dataset downloader script
        print("🚀 Running dataset downloader...")
        result = subprocess.run([sys.executable, "download_celeb_df.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dataset downloaded successfully!")
            return True
        else:
            print(f"❌ Dataset download failed: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ Dataset downloader script not found!")
        print("Please make sure 'download_celeb_df.py' is in the current directory")
        return False

def train_model():
    """Train the deepfake detection model"""
    print("\n🏋️  Starting model training...")
    
    # Check if dataset exists
    if not os.path.exists("data/celeb_df/splits"):
        print("❌ Dataset not found! Please download the dataset first.")
        return False
    
    try:
        print("🚀 Starting training pipeline...")
        result = subprocess.run([sys.executable, "complete_training_pipeline.py"], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("✅ Model training completed!")
            return True
        else:
            print("❌ Training failed!")
            return False
            
    except FileNotFoundError:
        print("❌ Training script not found!")
        return False

def run_webcam_detection():
    """Run real-time webcam detection"""
    print("\n🎥 Starting real-time webcam detection...")
    
    # Check if model exists
    if not os.path.exists("best_deepfake_detector.pth"):
        print("❌ Trained model not found! Please train the model first.")
        return False
    
    try:
        print("🚀 Starting webcam detection...")
        print("Press 'q' to quit")
        result = subprocess.run([sys.executable, "enhanced_deepfake_detector.py", "--mode", "realtime"])
        
        return True
        
    except FileNotFoundError:
        print("❌ Detection script not found!")
        return False

def launch_web_interface():
    """Launch the Streamlit web interface"""
    print("\n🌐 Launching web interface...")
    
    try:
        print("🚀 Starting Streamlit app...")
        print("🌐 The web interface will open in your browser")
        print("📱 Access it at: http://localhost:8501")
        
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_deepfake_app.py"])
        
    except FileNotFoundError:
        print("❌ Streamlit app not found!")
        return False

def test_single_image():
    """Test detection on a single image"""
    print("\n🧪 Single image testing...")
    
    # Check if model exists
    if not os.path.exists("best_deepfake_detector.pth"):
        print("❌ Trained model not found! Please train the model first.")
        return False
    
    image_path = input("📸 Enter image path: ").strip()
    
    if not os.path.exists(image_path):
        print("❌ Image not found!")
        return False
    
    try:
        result = subprocess.run([sys.executable, "enhanced_deepfake_detector.py", 
                               "--mode", "test", "--image", image_path])
        return True
        
    except FileNotFoundError:
        print("❌ Detection script not found!")
        return False

def view_results():
    """View training results and plots"""
    print("\n📊 Viewing training results...")
    
    results_dir = "results"
    if not os.path.exists(results_dir):
        print("❌ No results found! Please train the model first.")
        return False
    
    # List available result files
    result_files = os.listdir(results_dir)
    if not result_files:
        print("❌ No result files found!")
        return False
    
    print("📁 Available result files:")
    for i, file in enumerate(result_files, 1):
        print(f"  {i}. {file}")
    
    # Open results directory
    try:
        if sys.platform == "win32":
            os.startfile(results_dir)
        elif sys.platform == "darwin":
            subprocess.run(["open", results_dir])
        else:
            subprocess.run(["xdg-open", results_dir])
        
        print(f"✅ Results directory opened: {os.path.abspath(results_dir)}")
        
    except:
        print(f"📁 Results location: {os.path.abspath(results_dir)}")

def show_help():
    """Show help and documentation"""
    print("\n❓ HELP AND DOCUMENTATION")
    print("=" * 50)
    
    print("""
📖 QUICK START GUIDE:

1. 🔧 SETUP:
   - Run this script and choose option 1 to download dataset
   - Ensure you have Kaggle API set up for dataset download

2. 🏋️  TRAINING:
   - Choose option 2 to train the model
   - Training takes 10-20 minutes depending on your hardware
   - Results are saved in 'results/' directory

3. 🎥 REAL-TIME DETECTION:
   - Choose option 3 for webcam detection
   - Make sure you have a working webcam
   - Press 'q' to quit detection

4. 🌐 WEB INTERFACE:
   - Choose option 4 to launch the Streamlit app
   - Upload images, batch process, view analytics
   - Access at http://localhost:8501

📋 PROJECT STRUCTURE:
├── data/                    # Dataset files
├── checkpoints/            # Model checkpoints
├── results/               # Training results and plots
├── logs/                  # Training logs
├── enhanced_deepfake_detector.py    # Main detection script
├── complete_training_pipeline.py   # Training pipeline
├── streamlit_deepfake_app.py       # Web interface
└── download_celeb_df.py           # Dataset downloader

🛠️ TROUBLESHOOTING:
- Ensure Python 3.7+ is installed
- Install CUDA drivers for GPU acceleration
- Check webcam permissions for real-time detection
- Verify Kaggle API setup for dataset download

📚 ASSESSMENT COMPONENTS:
✅ Data Acquisition - Automated dataset download
✅ Processing - Advanced augmentation and preprocessing  
✅ Model Development - Enhanced CNN architecture
✅ Optimization - Learning rate scheduling, early stopping
✅ Real-time Detection - OpenCV integration
✅ Interactive Demo - Streamlit web interface

🎯 EXPECTED PERFORMANCE:
- Training Accuracy: 90-95%
- Validation Accuracy: 85-92%
- Real-time Processing: ~15-30 FPS
- Single Image: ~0.1 seconds
    """)

def main():
    """Main menu loop"""
    print_banner()
    
    # Initial setup check
    if not check_python_version():
        return
    
    # Ask if user wants to install requirements
    install_deps = input("\n🔧 Install required packages? (y/n): ").lower().strip()
    if install_deps in ['y', 'yes']:
        install_requirements()
    
    # Create project structure
    create_project_structure()
    
    # Main menu loop
    while True:
        show_usage_menu()
        
        try:
            choice = input("\n🎯 Enter your choice (1-8): ").strip()
            
            if choice == '1':
                download_dataset()
                
            elif choice == '2':
                train_model()
                
            elif choice == '3':
                run_webcam_detection()
                
            elif choice == '4':
                launch_web_interface()
                
            elif choice == '5':
                test_single_image()
                
            elif choice == '6':
                view_results()
                
            elif choice == '7':
                show_help()
                
            elif choice == '8':
                print("\n👋 Thanks for using the Deepfake Detection System!")
                print("🎓 Good luck with your assessment!")
                break
                
            else:
                print("❌ Invalid choice! Please enter 1-8.")
            
            # Pause before showing menu again
            input("\n⏸️  Press Enter to continue...")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            input("⏸️  Press Enter to continue...")

if __name__ == "__main__":
    main()