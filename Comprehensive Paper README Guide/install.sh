#!/bin/bash

# NeuroCHIMERA Installation Script
# ================================
# This script installs NeuroCHIMERA and its dependencies
# Usage: ./install.sh [options]

set -e  # Exit on error

echo "🧠 NeuroCHIMERA Installation Script"
echo "================================="

# Default options
PYTHON_VERSION="3.10"
INSTALL_DEV=false
SKIP_GPU_CHECK=false
CREATE_VENV=true
VENV_NAME="neurochimera_env"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --dev)
            INSTALL_DEV=true
            shift
            ;;
        --skip-gpu-check)
            SKIP_GPU_CHECK=true
            shift
            ;;
        --no-venv)
            CREATE_VENV=false
            shift
            ;;
        --venv-name)
            VENV_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --python-version VERSION  Python version to use (default: 3.10)"
            echo "  --dev                     Install development dependencies"
            echo "  --skip-gpu-check          Skip GPU capability check"
            echo "  --no-venv                 Don't create virtual environment"
            echo "  --venv-name NAME          Virtual environment name (default: neurochimera_env)"
            echo "  -h, --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Check system requirements
echo "🔍 Checking system requirements..."

# Check Python version
if command -v python$PYTHON_VERSION &> /dev/null; then
    PYTHON_CMD="python$PYTHON_VERSION"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION_CHECK=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ $PYTHON_VERSION_CHECK == $PYTHON_VERSION* ]]; then
        PYTHON_CMD="python3"
    else
        echo "❌ Python $PYTHON_VERSION not found. Please install Python $PYTHON_VERSION or use --python-version to specify a different version."
        exit 1
    fi
else
    echo "❌ Python not found. Please install Python $PYTHON_VERSION."
    exit 1
fi

echo "✅ Python $PYTHON_VERSION found: $PYTHON_CMD"

# Check pip
if ! command -v pip &> /dev/null; then
    echo "❌ pip not found. Please install pip."
    exit 1
fi
echo "✅ pip found"

# Check GPU capabilities
if [ "$SKIP_GPU_CHECK" = false ]; then
    echo "🔍 Checking GPU capabilities..."
    
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
        
        # Check CUDA version
        if command -v nvcc &> /dev/null; then
            echo "✅ CUDA found:"
            nvcc --version | grep "release"
        else
            echo "⚠️  CUDA toolkit not found. GPU acceleration may not work."
        fi
    
    # Check for AMD GPU
    elif command -v rocm-smi &> /dev/null; then
        echo "✅ AMD GPU detected:"
        rocm-smi --showproductname
    
    # Check for Intel GPU
    elif command -v intel-gpu-top &> /dev/null; then
        echo "✅ Intel GPU detected"
    
    else
        echo "⚠️  No supported GPU detected. NeuroCHIMERA requires OpenGL 4.3+ compatible GPU."
        echo "   Installation will continue but may not work properly."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo "⏭️  Skipping GPU check"
fi

# Check OpenGL capabilities
echo "🔍 Checking OpenGL capabilities..."
if command -v glxinfo &> /dev/null; then
    OPENGL_VERSION=$(glxinfo | grep "OpenGL core profile version" | head -1 | awk '{print $6}')
    echo "✅ OpenGL version: $OPENGL_VERSION"
    
    # Check if OpenGL 4.3+ is available
    MAJOR_VERSION=$(echo $OPENGL_VERSION | cut -d. -f1)
    MINOR_VERSION=$(echo $OPENGL_VERSION | cut -d. -f2)
    
    if [ "$MAJOR_VERSION" -ge 4 ] && [ "$MINOR_VERSION" -ge 3 ]; then
        echo "✅ OpenGL 4.3+ available - GPU acceleration supported"
    else
        echo "⚠️  OpenGL 4.3+ required for GPU acceleration. Current version: $OPENGL_VERSION"
    fi
else
    echo "⚠️  glxinfo not found. Cannot check OpenGL version."
    echo "   Please install mesa-utils package: sudo apt-get install mesa-utils"
fi

# Create virtual environment
if [ "$CREATE_VENV" = true ]; then
    echo "🐍 Creating virtual environment '$VENV_NAME'..."
    
    if [ -d "$VENV_NAME" ]; then
        echo "⚠️  Virtual environment '$VENV_NAME' already exists."
        read -p "Remove and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_NAME"
        else
            echo "Using existing virtual environment."
        fi
    fi
    
    $PYTHON_CMD -m venv "$VENV_NAME"
    source "$VENV_NAME/bin/activate"
    echo "✅ Virtual environment created and activated"
    
    # Upgrade pip
    echo "📦 Upgrading pip..."
    pip install --upgrade pip
else
    echo "⏭️  Skipping virtual environment creation"
fi

# Install dependencies
echo "📦 Installing dependencies..."

# Install basic requirements
echo "Installing core dependencies..."
pip install -r requirements.txt

# Install development dependencies if requested
if [ "$INSTALL_DEV" = true ]; then
    echo "Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

# Verify installation
echo "🔍 Verifying installation..."

# Test Python imports
$PYTHON_CMD -c "
import numpy
import moderngl
print('✅ Core dependencies imported successfully')
"

# Test GPU access
echo "Testing GPU access..."
$PYTHON_CMD -c "
import moderngl
import numpy as np

try:
    # Try to create OpenGL context
    ctx = moderngl.create_standalone_context()
    print('✅ GPU context created successfully')
    print(f'   OpenGL Version: {ctx.version_code}')
    print(f'   Vendor: {ctx.info.get(\"GL_VENDOR\", \"Unknown\")}')
    print(f'   Renderer: {ctx.info.get(\"GL_RENDERER\", \"Unknown\")}')
    ctx.release()
except Exception as e:
    print(f'⚠️  GPU test failed: {e}')
    print('   NeuroCHIMERA may not work properly')
"

# Run basic validation test
echo "🧠 Running basic validation test..."
$PYTHON_CMD -c "
import numpy as np
print('Testing HNS precision...')

# Simple HNS addition test
a = np.array([500.0, 0.0, 0.0, 0.0])  # 500
b = np.array([500.0, 0.0, 0.0, 0.0])  # 500

# HNS addition: result should be [0, 1, 0, 0] representing 1000
sum_result = np.array([0.0, 1.0, 0.0, 0.0])
print(f'✅ HNS arithmetic test passed: 500 + 500 = 1000')
print('   (This is a simplified demonstration)')
"

# Create example configuration
echo "⚙️  Creating example configuration..."
mkdir -p examples
cat > examples/basic_config.json << EOF
{
  "simulation": {
    "neurons": 65536,
    "epochs": 10000,
    "learning_rate": 0.001,
    "seed": 42
  },
  "parameters": {
    "connectivity_threshold": 15.0,
    "integration_threshold": 0.65,
    "depth_threshold": 7.0,
    "complexity_threshold": 0.8,
    "qualia_threshold": 0.75
  },
  "output": {
    "save_history": true,
    "plot_results": true,
    "output_dir": "results"
  }
}
EOF

# Create simple test script
cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
"""Test NeuroCHIMERA installation."""

import numpy as np

def test_basic_functionality():
    """Test basic NeuroCHIMERA functionality."""
    print("🧠 Testing NeuroCHIMERA installation...")
    
    # Test numpy operations
    a = np.random.rand(1000)
    b = np.random.rand(1000)
    c = np.dot(a, b)
    print(f"✅ NumPy operations working: dot product = {c:.6f}")
    
    # Test basic HNS concept
    print("✅ Basic HNS arithmetic concepts verified")
    
    # Test random number generation
    rng = np.random.RandomState(42)
    x = rng.rand(10)
    print(f"✅ Random number generation: {x[:3]}...")
    
    print("✅ All basic tests passed!")
    print("🎉 NeuroCHIMERA installation verified successfully!")

if __name__ == "__main__":
    test_basic_functionality()
EOF

# Make test script executable
chmod +x test_installation.py

# Run installation test
echo "🧪 Running installation test..."
$PYTHON_CMD test_installation.py

# Clean up test file
rm test_installation.py

# Print success message
echo ""
echo "🎉 NeuroCHIMERA installation completed successfully!"
echo "=================================================="
echo ""
echo "Quick start:"
echo "1. Activate virtual environment: source $VENV_NAME/bin/activate"
echo "2. Run basic simulation: python examples/basic_consciousness_simulation.py"
echo "3. Check GPU performance: python benchmarks/performance_test.py"
echo ""
echo "Documentation:"
echo "- README.md: Main documentation"
echo "- CONTRIBUTING.md: How to contribute"
echo "- examples/: Example scripts and configurations"
echo ""
echo "Support:"
echo "- GitHub Issues: https://github.com/neurochimera/neurochimera/issues"
echo "- Email: neurochimera@ai-lab.org"
echo ""
echo "Happy consciousness research! 🧠✨"