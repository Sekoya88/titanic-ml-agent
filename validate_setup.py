#!/usr/bin/env python3
"""
Validation script to check if the Titanic ML Agent setup is correct
"""

import os
import sys

def check_files():
    """Check if all required files exist"""
    required_files = [
        'titanic_agent.py',
        'train_agent.py',
        'requirements.txt',
        'README.md',
        'data/README.md'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    else:
        print("✅ All required files present")
        return True

def check_data_files():
    """Check if data files are present"""
    data_files = ['data/train.csv', 'data/test.csv']
    missing_data = []
    
    for file in data_files:
        if not os.path.exists(file):
            missing_data.append(file)
    
    if missing_data:
        print("⚠️  Missing data files (download from Kaggle):")
        for file in missing_data:
            print(f"  - {file}")
        return False
    else:
        print("✅ Data files present")
        return True

def check_dependencies():
    """Check if dependencies can be imported"""
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'sklearn': 'scikit-learn',
        'joblib': 'joblib'
    }
    
    missing_packages = []
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ Missing packages (run: pip install -r requirements.txt):")
        for package in missing_packages:
            print(f"  - {package}")
        return False
    else:
        print("✅ All required packages installed")
        return True

def main():
    print("🔍 Validating Titanic ML Agent Setup")
    print("=" * 40)
    
    files_ok = check_files()
    data_ok = check_data_files()
    deps_ok = check_dependencies()
    
    print("=" * 40)
    
    if files_ok and deps_ok:
        if data_ok:
            print("🎉 Setup complete! Ready to run:")
            print("   python train_agent.py")
        else:
            print("⚠️  Setup almost complete!")
            print("   1. Download train.csv and test.csv from Kaggle")
            print("   2. Place them in the data/ directory")
            print("   3. Run: python train_agent.py")
    else:
        print("❌ Setup incomplete. Fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 