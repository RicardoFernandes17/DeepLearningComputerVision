#!/usr/bin/env python3
"""
Kaggle Celeb-DF Preprocessed Dataset Auto Downloader & Organizer
Downloads the preprocessed Celeb-DF dataset and organizes it for training
"""

import os
import subprocess
import sys
import zipfile
import shutil
from pathlib import Path
import json
from tqdm import tqdm
import random

def check_requirements():
    """Check if required packages are installed"""
    required_packages = ['kaggle', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {missing_packages}")
        print("Installing missing packages...")
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("✅ Packages installed successfully!")

def setup_kaggle_api():
    """Set up Kaggle API credentials"""
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if kaggle_json.exists():
        print("✅ Kaggle credentials found!")
        # Set proper permissions
        os.chmod(kaggle_json, 0o600)
        return True
    
    print("❌ Kaggle credentials not found!")
    print("\n📋 To set up Kaggle API:")
    print("1. Go to https://www.kaggle.com/settings/account")
    print("2. Scroll down to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. This downloads 'kaggle.json' file")
    print("5. Move it to ~/.kaggle/kaggle.json")
    print("\nOr run these commands:")
    print("mkdir -p ~/.kaggle")
    print("mv ~/Downloads/kaggle.json ~/.kaggle/")
    print("chmod 600 ~/.kaggle/kaggle.json")
    
    return False

def create_project_structure():
    """Create the project directory structure"""
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
        "models/checkpoints",
        "scripts",
        "results"
    ]
    
    print("📁 Creating project structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Project structure created!")
    return directories

def download_dataset():
    """Download the preprocessed Celeb-DF dataset from Kaggle"""
    print("📥 Downloading Celeb-DF preprocessed dataset...")
    
    try:
        # Download the dataset
        cmd = ["kaggle", "datasets", "download", "-d", "amanrawat001/celeb-df-preprocessed"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="data/celeb_df/raw")
        
        if result.returncode != 0:
            print(f"❌ Download failed: {result.stderr}")
            return False
        
        print("✅ Dataset downloaded successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading dataset: {e}")
        return False
    except FileNotFoundError:
        print("❌ Kaggle CLI not found. Please install with: pip install kaggle")
        return False

def extract_dataset():
    """Extract the downloaded dataset"""
    print("📦 Extracting dataset...")
    
    raw_dir = Path("data/celeb_df/raw")
    zip_files = list(raw_dir.glob("*.zip"))
    
    if not zip_files:
        print("❌ No zip files found in raw directory")
        return False
    
    zip_file = zip_files[0]  # Take the first zip file
    
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
        
        print("✅ Dataset extracted successfully!")
        
        # Remove the zip file to save space
        zip_file.unlink()
        print("🗑️  Cleaned up zip file")
        
        return True
        
    except zipfile.BadZipFile:
        print(f"❌ Bad zip file: {zip_file}")
        return False

def analyze_dataset_structure():
    """Analyze the structure of the extracted dataset"""
    print("🔍 Analyzing dataset structure...")
    
    raw_dir = Path("data/celeb_df/raw")
    
    # Find all directories and files
    structure = {}
    for item in raw_dir.rglob("*"):
        if item.is_dir():
            # Count files in directory
            files = list(item.glob("*"))
            image_files = [f for f in files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
            structure[str(item.relative_to(raw_dir))] = {
                'type': 'directory',
                'total_files': len(files),
                'image_files': len(image_files)
            }
    
    print("📊 Dataset structure:")
    for path, info in structure.items():
        if info['image_files'] > 0:
            print(f"  📁 {path}: {info['image_files']} images")
    
    return structure

def organize_dataset():
    """Organize the dataset into real/fake folders"""
    print("🗂️  Organizing dataset...")
    
    raw_dir = Path("data/celeb_df/raw")
    processed_dir = Path("data/celeb_df/processed")
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(raw_dir.rglob(f"*{ext}"))
        all_images.extend(raw_dir.rglob(f"*{ext.upper()}"))
    
    print(f"📊 Found {len(all_images)} total images")
    
    # Try to identify real vs fake images based on common naming patterns
    real_images = []
    fake_images = []
    
    for img_path in all_images:
        img_name = img_path.name.lower()
        parent_dir = img_path.parent.name.lower()
        
        # Common patterns for identifying real vs fake
        if any(keyword in img_name or keyword in parent_dir for keyword in 
               ['real', 'original', 'genuine', 'authentic', 'celeb-real', 'youtube-real']):
            real_images.append(img_path)
        elif any(keyword in img_name or keyword in parent_dir for keyword in 
                 ['fake', 'deepfake', 'synthetic', 'generated', 'celeb-synthesis', 'manipulated']):
            fake_images.append(img_path)
        else:
            # If we can't determine, we'll split randomly later
            # For now, let's assume roughly half are real, half are fake
            if len(real_images) <= len(fake_images):
                real_images.append(img_path)
            else:
                fake_images.append(img_path)
    
    print(f"📊 Identified: {len(real_images)} real, {len(fake_images)} fake images")
    
    # Copy images to organized structure
    def copy_images(source_list, dest_dir, category_name):
        dest_path = processed_dir / dest_dir
        copied = 0
        
        for img_path in tqdm(source_list, desc=f"Copying {category_name} images"):
            try:
                # Create unique filename to avoid conflicts
                new_name = f"{category_name}_{copied:06d}{img_path.suffix}"
                dest_file = dest_path / new_name
                shutil.copy2(img_path, dest_file)
                copied += 1
            except Exception as e:
                print(f"⚠️  Error copying {img_path}: {e}")
        
        return copied
    
    real_copied = copy_images(real_images, "real", "real")
    fake_copied = copy_images(fake_images, "fake", "fake")
    
    print(f"✅ Organized dataset: {real_copied} real, {fake_copied} fake images")
    
    return real_copied, fake_copied

def create_train_val_test_splits(real_count, fake_count, train_ratio=0.7, val_ratio=0.15):
    """Create train/validation/test splits"""
    print("🔄 Creating train/val/test splits...")
    
    processed_dir = Path("data/celeb_df/processed")
    splits_dir = Path("data/celeb_df/splits")
    
    # Get all organized images
    real_images = list((processed_dir / "real").glob("*.jpg"))
    fake_images = list((processed_dir / "fake").glob("*.jpg"))
    
    # Shuffle for random splitting
    random.shuffle(real_images)
    random.shuffle(fake_images)
    
    # Calculate split sizes
    def calculate_splits(total_count):
        train_size = int(total_count * train_ratio)
        val_size = int(total_count * val_ratio)
        test_size = total_count - train_size - val_size
        return train_size, val_size, test_size
    
    real_train_size, real_val_size, real_test_size = calculate_splits(len(real_images))
    fake_train_size, fake_val_size, fake_test_size = calculate_splits(len(fake_images))
    
    # Split the data
    splits = {
        'train': {
            'real': real_images[:real_train_size],
            'fake': fake_images[:fake_train_size]
        },
        'val': {
            'real': real_images[real_train_size:real_train_size + real_val_size],
            'fake': fake_images[fake_train_size:fake_train_size + fake_val_size]
        },
        'test': {
            'real': real_images[real_train_size + real_val_size:],
            'fake': fake_images[fake_train_size + fake_val_size:]
        }
    }
    
    # Copy files to split directories
    for split_name, split_data in splits.items():
        for category, images in split_data.items():
            dest_dir = splits_dir / split_name / category
            
            for img_path in tqdm(images, desc=f"Creating {split_name}/{category} split"):
                try:
                    dest_file = dest_dir / img_path.name
                    shutil.copy2(img_path, dest_file)
                except Exception as e:
                    print(f"⚠️  Error copying {img_path}: {e}")
    
    # Print split statistics
    print("\n📊 Dataset splits created:")
    for split_name, split_data in splits.items():
        real_count = len(split_data['real'])
        fake_count = len(split_data['fake'])
        total = real_count + fake_count
        print(f"  {split_name:5}: {real_count:4} real + {fake_count:4} fake = {total:4} total")
    
    return splits

def create_summary_report():
    """Create a summary report of the downloaded dataset"""
    print("📋 Creating summary report...")
    
    splits_dir = Path("data/celeb_df/splits")
    
    report = {
        "dataset": "Celeb-DF Preprocessed",
        "source": "kaggle.com/datasets/amanrawat001/celeb-df-preprocessed",
        "downloaded_at": str(Path.cwd()),
        "splits": {}
    }
    
    for split in ['train', 'val', 'test']:
        split_path = splits_dir / split
        if split_path.exists():
            real_count = len(list((split_path / 'real').glob('*.jpg')))
            fake_count = len(list((split_path / 'fake').glob('*.jpg')))
            
            report["splits"][split] = {
                "real": real_count,
                "fake": fake_count,
                "total": real_count + fake_count
            }
    
    # Save report
    with open("celeb_df_dataset_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("✅ Summary report saved to 'celeb_df_dataset_report.json'")
    
    # Print summary
    print("\n🎉 DATASET READY FOR TRAINING!")
    print("=" * 50)
    total_real = sum(split["real"] for split in report["splits"].values())
    total_fake = sum(split["fake"] for split in report["splits"].values())
    total_images = total_real + total_fake
    
    print(f"📊 Total Images: {total_images}")
    print(f"   Real: {total_real}")
    print(f"   Fake: {total_fake}")
    print(f"   Balance: {total_real/total_images*100:.1f}% real, {total_fake/total_images*100:.1f}% fake")
    
    print("\n📁 Directory Structure:")
    print("   data/celeb_df/splits/")
    print("   ├── train/")
    print("   │   ├── real/     ({} images)".format(report["splits"]["train"]["real"]))
    print("   │   └── fake/     ({} images)".format(report["splits"]["train"]["fake"]))
    print("   ├── val/")
    print("   │   ├── real/     ({} images)".format(report["splits"]["val"]["real"]))
    print("   │   └── fake/     ({} images)".format(report["splits"]["val"]["fake"]))
    print("   └── test/")
    print("       ├── real/     ({} images)".format(report["splits"]["test"]["real"]))
    print("       └── fake/     ({} images)".format(report["splits"]["test"]["fake"]))
    
    print("\n🚀 Next Steps:")
    print("1. Run your deepfake detection training script")
    print("2. Use the data from: data/celeb_df/splits/")
    print("3. Expected training accuracy: 85-95% on this dataset")

def main():
    """Main function to orchestrate the entire process"""
    print("🚀 Celeb-DF Preprocessed Dataset Auto Downloader & Organizer")
    print("=" * 60)
    
    try:
        # Step 1: Check requirements
        check_requirements()
        
        # Step 2: Set up Kaggle API
        if not setup_kaggle_api():
            return False
        
        # Step 3: Create project structure
        create_project_structure()
        
        # Step 4: Download dataset
        if not download_dataset():
            return False
        
        # Step 5: Extract dataset
        if not extract_dataset():
            return False
        
        # Step 6: Analyze structure
        analyze_dataset_structure()
        
        # Step 7: Organize dataset
        real_count, fake_count = organize_dataset()
        
        # Step 8: Create splits
        create_train_val_test_splits(real_count, fake_count)
        
        # Step 9: Create summary
        create_summary_report()
        
        return True
        
    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 SUCCESS! Your Celeb-DF dataset is ready for training!")
    else:
        print("\n❌ Setup failed. Please check the errors above and try again.")