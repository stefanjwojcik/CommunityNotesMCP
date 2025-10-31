@echo off
REM Installation script for Community Notes MCP Server (Windows)

REM Parse arguments
set DEMO_MODE=false
if "%1"=="--demo" set DEMO_MODE=true

echo ======================================
echo Community Notes MCP Server - Setup
if "%DEMO_MODE%"=="true" echo (DEMO MODE - 1000 records only)
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

REM Check if data files exist, download if missing
echo Checking for data files...
if not exist "data" mkdir data

set NOTES_URL=https://ton.twimg.com/birdwatch-public-data/2025/10/31/notes/notes-00000.zip
set STATUS_URL=https://ton.twimg.com/birdwatch-public-data/2025/10/31/noteStatusHistory/noteStatusHistory-00000.zip

if not exist "data\notes-00000.tsv" (
    echo notes-00000.tsv not found. Downloading from Twitter...
    curl -L -o "data\notes-00000.zip" "%NOTES_URL%"
    if errorlevel 1 (
        echo ERROR: Failed to download notes-00000.zip
        echo Please ensure curl is installed or manually download the file.
        exit /b 1
    )

    echo Extracting notes-00000.zip...
    tar -xf "data\notes-00000.zip" -C data
    if errorlevel 1 (
        echo ERROR: Failed to extract notes-00000.zip
        exit /b 1
    )
    del "data\notes-00000.zip"
    echo notes-00000.tsv extracted successfully!
) else (
    echo notes-00000.tsv found!
)

if not exist "data\noteStatusHistory-00000.tsv" (
    echo noteStatusHistory-00000.tsv not found. Downloading from Twitter...
    curl -L -o "data\noteStatusHistory-00000.zip" "%STATUS_URL%"
    if errorlevel 1 (
        echo ERROR: Failed to download noteStatusHistory-00000.zip
        echo Please ensure curl is installed or manually download the file.
        exit /b 1
    )

    echo Extracting noteStatusHistory-00000.zip...
    tar -xf "data\noteStatusHistory-00000.zip" -C data
    if errorlevel 1 (
        echo ERROR: Failed to extract noteStatusHistory-00000.zip
        exit /b 1
    )
    del "data\noteStatusHistory-00000.zip"
    echo noteStatusHistory-00000.tsv extracted successfully!
) else (
    echo noteStatusHistory-00000.tsv found!
)
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
if "%DEMO_MODE%"=="true" (
    echo DEMO MODE: Processing only 1000 records (~1-2 minutes^)
) else (
    echo This may take 15-30 minutes depending on your hardware.
    echo Processing ~2M records and generating embeddings...
)
echo ======================================
echo.

if "%DEMO_MODE%"=="true" (
    python ingest_data.py --demo
) else (
    python ingest_data.py
)

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
if "%DEMO_MODE%"=="true" (
    echo Database created: community_notes_demo.db
    echo (Demo database with 1000 records^)
) else (
    echo Database created: community_notes.db
)
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
