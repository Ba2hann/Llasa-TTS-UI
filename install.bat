@echo off
setlocal

:: === Configuration ===
set PYTHON_EXE=python
set VENV_DIR=.venv
set REQUIRED_PYTHON_VERSION=3.9
:: If 'python' is not Python 3.9, try changing PYTHON_EXE to:
:: set PYTHON_EXE=py -3.9

echo Starting setup process...
echo Using Python executable: %PYTHON_EXE%
echo Virtual environment directory: %VENV_DIR%
echo Required Python Version: %REQUIRED_PYTHON_VERSION%
echo.

:: === Check if Python exists and potentially check version (basic check) ===
%PYTHON_EXE% --version > nul 2>&1
if errorlevel 1 (
    echo ERROR: '%PYTHON_EXE%' command not found.
    echo Please ensure Python %REQUIRED_PYTHON_VERSION% is installed and added to your PATH.
    echo You might need to adjust PYTHON_EXE variable at the top of this script.
    pause
    exit /b 1
)
echo Found %PYTHON_EXE% executable. Note: Advanced version check not implemented, please ensure it's version %REQUIRED_PYTHON_VERSION%.
echo.

:: === Check if venv already exists ===
if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Virtual environment '%VENV_DIR%' already exists.
    echo Skipping creation. If you want to recreate it, delete the '%VENV_DIR%' folder manually and rerun setup.
) else (
    echo Creating virtual environment in '%VENV_DIR%'...
    %PYTHON_EXE% -m venv %VENV_DIR%
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment. Check Python installation and permissions.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
)
echo.

:: === Activate virtual environment ===
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)
echo Environment activated. Subsequent commands will run inside the venv.
echo.

:: === Install packages ===
echo Installing base audio and array libraries (librosa, numpy, xcodec2, audiofile)...
python -m pip install --upgrade pip setuptools wheel
pip install librosa numpy xcodec2 audiofile
if errorlevel 1 (
    echo ERROR: Failed to install base libraries. Check pip and network connection.
    pause
    exit /b 1
)
echo Base libraries installed.
echo.

echo Installing/Updating PyTorch with CUDA 12.6 support (torch, torchvision, torchaudio)...
echo Using index: https://download.pytorch.org/whl/cu126
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 -U --force-reinstall
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch.
    echo Common issues: Incompatible NVIDIA driver/CUDA version, network problems, insufficient permissions.
    echo You might need CUDA Toolkit 12.6 or compatible drivers.
    echo As noted in comments, you could try torch==2.5.0 if 2.6.0 fails.
    pause
    exit /b 1
)
echo PyTorch installed/updated.
echo.

echo Installing AI/ML acceleration and UI libraries (bitsandbytes, triton-windows, transformers, optimum-quanto, accelerate, gradio)...
pip install bitsandbytes triton-windows transformers==4.48.0 optimum-quanto accelerate gradio -U
if errorlevel 1 (
    echo ERROR: Failed to install AI/ML/UI libraries. Check pip and network connection.
    pause
    exit /b 1
)
echo AI/ML/UI libraries installed.
echo.

echo Installing specific Pydantic version (pydantic==2.10.6)...
pip install pydantic==2.10.6
if errorlevel 1 (
    echo ERROR: Failed to install Pydantic 2.10.6. Check pip and network connection.
    pause
    exit /b 1
)
echo Pydantic installed.
echo.

echo Setup complete!
echo You can now run the application using run.bat.
echo.
pause
endlocal