#!/bin/bash
# Quick install script - bypasses virtual environment issues

echo "🚀 Quick Titanic ML Agent Setup..."
echo "Installing packages globally (safe for this project)"

pip3 install -r requirements.txt --break-system-packages

echo "✅ Installation complete!"
echo ""
echo "🎯 Ready to run:"
echo "  python3 train_agent.py"
echo ""
echo "📊 Check setup:"
echo "  python3 validate_setup.py" 