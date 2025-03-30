@echo off
setlocal

:: === Configuration ===
set VENV_DIR=.venv
set PYTHON_SCRIPT=llasaUI.py

echo Starting application runner...
echo Virtual environment directory: %VENV_DIR%
echo Python script to run: %PYTHON_SCRIPT%
echo.

:: === Check if venv exists ===
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found in '%VENV_DIR%'.
    echo Please run setup.bat first to create the environment and install dependencies.
    pause
    exit /b 1
)

:: === Check if Python script exists ===
if not exist "%PYTHON_SCRIPT%" (
    echo ERROR: Python script '%PYTHON_SCRIPT%' not found in the current directory.
    echo Please ensure the script is present.
    pause
    exit /b 1
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
echo Environment activated.
echo.

:: === Run the Python script ===
echo Launching %PYTHON_SCRIPT%...
echo If the script requires arguments, you may need to modify this batch file.
echo --- Script Output Starts Below ---
python "%PYTHON_SCRIPT%"
echo --- Script Output Ends Above ---
echo.

echo Script execution finished or interrupted.
pause
endlocal