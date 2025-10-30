@echo off
REM Installation script for Community Notes MCP Server (Windows)

echo ======================================
echo Community Notes MCP Server - Setup
echo ======================================
echo.

REM Check Python version (3.10+ required)
echo Checking for Python 3.10 or higher...
python --version 2>nul
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.10 or higher.
    echo Download from: https://www.python.org/downloads/
    exit /b 1
)

REM Check Python version meets requirement
python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>nul
if errorlevel 1 (
    echo ERROR: Python 3.10 or higher is required!
    python --version
    echo Please install Python 3.10+ from: https://www.python.org/downloads/
    exit /b 1
)

python --version
echo.

REM Check if data files exist
echo Checking for data files...
if not exist "data\notes-00000.tsv" (
    echo ERROR: data\notes-00000.tsv not found!
    echo Please ensure the data files are in the data\ directory.
    exit /b 1
)

if not exist "data\noteStatusHistory-00000.tsv" (
    echo ERROR: data\noteStatusHistory-00000.tsv not found!
    echo Please ensure the data files are in the data\ directory.
    exit /b 1
)
echo Data files found!
echo.

REM Create virtual environment
echo Creating virtual environment...
if exist "venv" (
    echo Virtual environment already exists. Skipping creation.
) else (
    python -m venv venv
    echo Virtual environment created.
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo Dependencies installed!
echo.

REM Run data ingestion
echo ======================================
echo Starting data ingestion...
echo This may take 15-30 minutes depending on your hardware.
echo Processing ~2M records and generating embeddings...
echo ======================================
echo.

python ingest_data.py

if errorlevel 1 (
    echo.
    echo ERROR: Data ingestion failed!
    echo Please check the error messages above.
    exit /b 1
)

echo.
echo ======================================
echo Installation Complete!
echo ======================================
echo.
echo Database created: community_notes.db
echo.
echo Next steps:
echo 1. Configure your MCP client (e.g., Claude Desktop)
echo.
echo For Claude Desktop, add to your config file:
echo    Config location: %%APPDATA%%\Claude\claude_desktop_config.json
echo.
echo    {
echo      "mcpServers": {
echo        "community-notes": {
echo          "command": "python",
echo          "args": ["%CD%\server.py"]
echo        }
echo      }
echo    }
echo.
echo 2. Restart Claude Desktop
echo 3. Start using the community notes tools!
echo.
echo To start the server manually:
echo   venv\Scripts\activate.bat
echo   python server.py
echo.

pause
