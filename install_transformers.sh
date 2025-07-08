#!/bin/bash
# install_transformers.sh
# Script to install transformer dependencies for enhanced sentiment analysis

echo "🚀 Installing Transformer Dependencies for Enhanced Sentiment Analysis"
echo "======================================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 not found. Please install pip."
    exit 1
fi

echo "📦 Installing core transformer libraries..."

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "🐍 Detected Python version: $PYTHON_VERSION"

# Install PyTorch with appropriate method based on Python version
if [[ "$PYTHON_VERSION" == "3.13" ]]; then
    echo "⚠️  Python 3.13 detected - using nightly build or CPU-only version"
    echo "📦 Installing CPU-only PyTorch (recommended for Python 3.13)..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || {
        echo "❌ Failed to install PyTorch CPU version"
        echo "💡 Trying alternative installation method..."
        
        # Try nightly build as fallback
        pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu || {
            echo "❌ Failed to install PyTorch nightly build"
            echo "💡 As a last resort, trying to install without torch (transformers only)..."
            pip3 install transformers>=4.21.0 || {
                echo "❌ Failed to install transformers"
                exit 1
            }
            echo "⚠️  Warning: PyTorch installation failed. Some models may not work."
            echo "   You may need to downgrade Python or wait for PyTorch 3.13 support."
            return 0
        }
    }
    
    echo "📦 Installing transformers..."
    pip3 install transformers>=4.21.0 || {
        echo "❌ Failed to install transformers"
        exit 1
    }
    
else
    echo "📦 Installing standard PyTorch and transformers..."
    # For Python 3.8-3.12, use standard installation
    pip3 install torch>=1.12.0 || {
        echo "❌ Failed to install PyTorch"
        echo "💡 Trying CPU-only version..."
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || {
            echo "❌ Failed to install PyTorch CPU version"
            echo "💡 Visit: https://pytorch.org/get-started/locally/"
            exit 1
        }
    }
    
    pip3 install transformers>=4.21.0 || {
        echo "❌ Failed to install transformers"
        exit 1
    }
fi

echo "🎯 Installing additional dependencies..."

# Install additional helpful packages
pip3 install sentencepiece tokenizers accelerate || {
    echo "⚠️  Warning: Some additional packages failed to install"
    echo "   This may affect some specific models but core functionality should work"
}

echo "✅ Installation completed!"
echo ""
echo "🧪 Testing installation..."

# Test if transformers can be imported
python3 -c "
import transformers
import torch
print('✅ Transformers version:', transformers.__version__)
print('✅ PyTorch version:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
" || {
    echo "❌ Installation test failed"
    exit 1
}

echo ""
echo "🎉 Transformer dependencies installed successfully!"
echo ""
echo "📝 Next steps:"
echo "1. Run the test script: python3 test_transformer_sentiment.py"
echo "2. The first run will download models (~1-2GB) - this may take a few minutes"
echo "3. Models are cached locally for faster subsequent runs"
echo ""
echo "🔧 Available models:"
echo "- FinBERT: Specialized for financial sentiment analysis"
echo "- RoBERTa: General-purpose sentiment analysis"
echo "- Emotion Detection: Identifies emotions in text"
echo "- News Classification: Categorizes news types"
echo ""
echo "💡 Models will be downloaded automatically when first used"
echo "   Total download size: ~1-2GB depending on models used"
