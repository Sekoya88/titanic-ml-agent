#!/bin/bash
# Setup script for Titanic ML Agent

echo "🚀 Setting up Titanic ML Agent..."

# Remove existing venv if corrupted
if [ -d "venv" ] && [ ! -f "venv/bin/activate" ]; then
    echo "🗑️ Removing corrupted virtual environment..."
    rm -rf venv
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv --system-site-packages
    
    # Check if creation was successful
    if [ ! -f "venv/bin/activate" ]; then
        echo "❌ Failed to create virtual environment"
        echo "Trying alternative method..."
        python3 -m venv venv
        
        if [ ! -f "venv/bin/activate" ]; then
            echo "❌ Virtual environment creation failed"
            echo "Installing packages globally with --break-system-packages"
            pip3 install -r requirements.txt --break-system-packages
            echo "✅ Packages installed globally!"
            echo ""
            echo "You can now run:"
            echo "  python3 train_agent.py"
            exit 0
        fi
    fi
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip first
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "⬇️ Installing requirements..."
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "To activate the environment in the future:"
echo "  source venv/bin/activate"
echo ""
echo "To run the agent:"
echo "  python train_agent.py" 