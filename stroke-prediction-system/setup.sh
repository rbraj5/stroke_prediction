#!/bin/bash

echo "🏥 Stroke Prediction System - Setup Script"
echo "==========================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "✓ Virtual environment created"
echo ""
echo "🔧 Activating virtual environment..."

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo "✓ Virtual environment activated"
echo ""
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✓ Dependencies installed"
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models logs

echo "✓ Directories created"
echo ""

# Train models
echo "🤖 Training models..."
cd ml
python train.py
cd ..

echo "✓ Models trained"
echo ""

# Run tests
echo "🧪 Running tests..."
pytest tests/ -v

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To start the application:"
echo ""
echo "  1. Start API server:"
echo "     uvicorn api.main:app --reload"
echo ""
echo "  2. Start dashboard (in another terminal):"
echo "     streamlit run dashboard/app.py"
echo ""
echo "  3. Or use Docker:"
echo "     docker-compose up"
echo ""
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🎨 Dashboard: http://localhost:8501"
echo ""
