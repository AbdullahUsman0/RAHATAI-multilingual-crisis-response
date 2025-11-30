"""
STEP 1.2: CHECK YOUR ENVIRONMENT
This script checks which things are installed and which are missing.
"""

import sys
import subprocess

# Print section heading
print("=" * 70)
print("STEP 1.2: CHECKING YOUR ENVIRONMENT")
print("=" * 70)

# ------------------------------------------------------
# 1. CHECK PYTHON VERSION
# ------------------------------------------------------
print("\nPYTHON VERSION:")
python_version = sys.version
python_version_num = sys.version_info
print("   " + python_version)
if python_version_num.major == 3 and python_version_num.minor == 10:
    print("   ✓ Python 3.10 detected - DirectML training supported")
elif python_version_num.major == 3 and python_version_num.minor >= 8:
    print("   ⚠ Python 3.10 recommended for DirectML training")
    print("   Current version may not support tensorflow-directml-plugin")
else:
    print("   ⚠ Python 3.10 required for DirectML training")

# ------------------------------------------------------
# 2. CHECK REQUIRED PACKAGES
# ------------------------------------------------------
print("\nCHECKING REQUIRED PACKAGES:")
print("-" * 70)

# This is the list of packages we want to check
required_packages = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'scikit-learn': 'sklearn',
    'torch': 'torch',
    'tensorflow': 'tensorflow',
    'transformers': 'transformers',
    'sentence-transformers': 'sentence_transformers',
    'langchain': 'langchain',
    'faiss': 'faiss',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'nltk': 'nltk'
}

# Lists to store installed and missing packages
installed = []
missing = []

# Check each package one by one
for package_name, import_name in required_packages.items():
    try:
        __import__(import_name)
        print("   OK: " + package_name)
        installed.append(package_name)
    except ImportError:
        print("   MISSING: " + package_name + " - NOT INSTALLED")
        missing.append(package_name)

# ------------------------------------------------------
# 3. CHECK GPU AVAILABILITY (DirectML for AMD GPU)
# ------------------------------------------------------
print("\nHARDWARE CHECK:")
print("-" * 70)

# Check TensorFlow DirectML
try:
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices()
    print("   TensorFlow devices:", [str(d) for d in physical_devices])
    
    # Check for DirectML devices
    dml_devices = [d for d in physical_devices if 'DML' in str(d) or 'directml' in str(d).lower()]
    if dml_devices:
        print("   ✓ DirectML GPU detected - AMD GPU acceleration available")
        print(f"   DirectML devices: {len(dml_devices)}")
    else:
        print("   ⚠ DirectML not detected - check:")
        print("     1. Python 3.10 installed")
        print("     2. tensorflow-cpu==2.10.0 installed")
        print("     3. tensorflow-directml-plugin installed")
        print("     4. Latest AMD GPU drivers")
except Exception as e:
    print("   ⚠ Could not check TensorFlow DirectML:", str(e))

# Check PyTorch CUDA/DirectML
try:
    import torch

    # Check if CUDA GPU is available
    gpu_available = torch.cuda.is_available()

    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        print("   PyTorch CUDA GPU Available:", gpu_name)
        print("   CUDA Version:", cuda_version)
    else:
        print("   PyTorch: CUDA not available")
    
    # Check for torch-directml
    try:
        import torch_directml
        print("   ✓ torch-directml available for AMD GPU")
    except ImportError:
        print("   ⚠ torch-directml not available (PyTorch will use CPU)")
        
    # Check ONNX Runtime DirectML
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'DmlExecutionProvider' in providers:
            print("   ✓ ONNX Runtime DirectML available for AMD GPU")
        else:
            print("   ⚠ ONNX Runtime available but DirectML provider not found")
    except ImportError:
        print("   ⚠ ONNX Runtime not installed")
except:
    print("   ⚠ Could not check PyTorch GPU (PyTorch not installed)")

# ------------------------------------------------------
# 4. SUMMARY OF RESULTS
# ------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY:")
print("=" * 70)

# If some packages are missing
if len(missing) > 0:
    print("\nMissing", len(missing), "packages:")

    for pkg in missing:
        print("   - " + pkg)

    print("\nInstalled", len(installed), "packages")

    print("\n" + "=" * 70)
    print("NEXT STEP: Install missing packages")
    print("=" * 70)

    print("\nRun this command to install everything:")
    print("   pip install -r requirements.txt")

    print("\nOr install missing packages one by one:")
    for pkg in missing:
        print("   pip install " + pkg)

# If all packages are installed
else:
    print("\nAll required packages are installed!")
    print("\n" + "=" * 70)
    print("READY TO PROCEED TO NEXT STEP!")
    print("=" * 70)

# ------------------------------------------------------
# 5. FINAL INSTRUCTIONS
# ------------------------------------------------------
print("\n" + "=" * 70)
print("WHAT TO DO NEXT:")
print("=" * 70)

print("""
1. If packages are missing:
   - Run: pip install -r requirements.txt
   - Wait until installation finishes
   - Run this script again

2. If all packages are installed:
   - Go to Step 1.3: Prepare the combined dataset
""")

print("=" * 70)
