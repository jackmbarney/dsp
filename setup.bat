@echo off
REM Creates a virtual environment in .venv and installs dependencies from requirements.txt.

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: python was not found on PATH. Install Python and try again.
    exit /b 1
)

if not exist .venv (
    echo Creating virtual environment in .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: failed to create virtual environment.
        exit /b 1
    )
) else (
    echo .venv already exists, skipping creation.
)

echo Upgrading pip ...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 exit /b 1

echo Installing dependencies from requirements.txt ...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo.
echo Setup complete. Activate the environment with:
echo     .venv\Scripts\activate
