@echo off
REM Setup script for Python 3.10 with DirectML on Windows
REM This script helps set up the conda environment and activate it

echo ============================================================
echo Setting up Python 3.10 environment with DirectML support
echo ============================================================
echo.

REM Check if conda is available
where conda >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Conda is not found in PATH
    echo Please install Miniconda or Anaconda first
    echo Download from: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo Step 1: Creating conda environment (if it doesn't exist)...
conda create -n rahatai python=3.10 -y

echo.
echo Step 2: Activating environment...
REM Try to activate using the direct path method
call conda activate rahatai

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ============================================================
    echo IMPORTANT: Conda activation failed
    echo ============================================================
    echo.
    echo After running 'conda init', you need to:
    echo 1. Close this terminal window
    echo 2. Open a NEW terminal window
    echo 3. Run: conda activate rahatai
    echo.
    echo OR use this command in a NEW terminal:
    echo   %CONDA_PREFIX%\Scripts\activate.bat rahatai
    echo.
    pause
    exit /b 1
)

echo.
echo Step 3: Verifying Python version...
python --version

echo.
echo Step 4: Installing dependencies...
echo Installing TensorFlow 2.10 with DirectML...
pip install tensorflow-cpu==2.10.0
pip install tensorflow-directml-plugin

echo Installing ONNX Runtime with DirectML...
pip install onnxruntime-directml>=1.12.0
pip install onnx>=1.12.0
pip install onnxruntime-training>=1.12.0

echo Installing other requirements...
pip install -r requirements.txt

echo.
echo ============================================================
echo Setup complete!
echo ============================================================
echo.
echo To activate this environment in the future:
echo   conda activate rahatai
echo.
echo To verify DirectML is working:
echo   python RunScripts/check_environment.py
echo.
pause

