#!/bin/bash
# Installation script for Community Notes MCP Server

set -e  # Exit on error

echo "======================================"
echo "Community Notes MCP Server - Setup"
echo "======================================"
echo ""

# Find suitable Python version (3.10+)
echo "Checking for Python 3.10 or higher..."
PYTHON_CMD=""

for py_cmd in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v $py_cmd &> /dev/null; then
        version=$($py_cmd --version 2>&1 | awk '{print $2}')
        major=$(echo $version | cut -d. -f1)
        minor=$(echo $version | cut -d. -f2)

        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON_CMD=$py_cmd
            echo "Found $py_cmd (version $version)"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python 3.10 or higher is required!"
    echo "Please install Python 3.10+ and try again."
    echo ""
    echo "On macOS with Homebrew:"
    echo "  brew install python@3.11"
    echo ""
    exit 1
fi
echo ""

# Check if data files exist, download if missing
echo "Checking for data files..."
mkdir -p data

NOTES_URL="https://ton.twimg.com/birdwatch-public-data/2025/01/01/notes/notes-00000.tsv"
STATUS_URL="https://ton.twimg.com/birdwatch-public-data/2025/01/01/noteStatusHistory/noteStatusHistory-00000.tsv"

if [ ! -f "data/notes-00000.tsv" ]; then
    echo "notes-00000.tsv not found. Downloading from Twitter..."
    if command -v curl &> /dev/null; then
        curl -L -o "data/notes-00000.tsv" "$NOTES_URL"
    elif command -v wget &> /dev/null; then
        wget -O "data/notes-00000.tsv" "$NOTES_URL"
    else
        echo "ERROR: Neither curl nor wget found. Cannot download data files."
        echo "Please install curl or wget, or manually download files to data/ directory."
        exit 1
    fi

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to download notes-00000.tsv"
        exit 1
    fi
    echo "notes-00000.tsv downloaded successfully!"
else
    echo "notes-00000.tsv found!"
fi

if [ ! -f "data/noteStatusHistory-00000.tsv" ]; then
    echo "noteStatusHistory-00000.tsv not found. Downloading from Twitter..."
    if command -v curl &> /dev/null; then
        curl -L -o "data/noteStatusHistory-00000.tsv" "$STATUS_URL"
    elif command -v wget &> /dev/null; then
        wget -O "data/noteStatusHistory-00000.tsv" "$STATUS_URL"
    else
        echo "ERROR: Neither curl nor wget found. Cannot download data files."
        echo "Please install curl or wget, or manually download files to data/ directory."
        exit 1
    fi

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to download noteStatusHistory-00000.tsv"
        exit 1
    fi
    echo "noteStatusHistory-00000.tsv downloaded successfully!"
else
    echo "noteStatusHistory-00000.tsv found!"
fi
echo ""

# Create virtual environment
echo "Creating virtual environment with $PYTHON_CMD..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping creation."
else
    $PYTHON_CMD -m venv venv
    echo "Virtual environment created."
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "Dependencies installed!"
echo ""

# Run data ingestion
echo "======================================"
echo "Starting data ingestion..."
echo "This may take 15-30 minutes depending on your hardware."
echo "Processing ~2M records and generating embeddings..."
echo "======================================"
echo ""

python ingest_data.py

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "Installation Complete!"
    echo "======================================"
    echo ""
    echo "Database created: community_notes.db"
    echo ""
    echo "Next steps:"
    echo "1. Configure your MCP client (e.g., Claude Desktop)"
    echo ""
    echo "For Claude Desktop, add to your config file:"

    if [[ "$OSTYPE" == "darwin"* ]]; then
        config_path="~/Library/Application Support/Claude/claude_desktop_config.json"
    else
        config_path="%APPDATA%/Claude/claude_desktop_config.json"
    fi

    echo "   Config location: $config_path"
    echo ""
    echo '   {'
    echo '     "mcpServers": {'
    echo '       "community-notes": {'
    echo '         "command": "python",'
    echo "         \"args\": [\"$(pwd)/server.py\"]"
    echo '       }'
    echo '     }'
    echo '   }'
    echo ""
    echo "2. Restart Claude Desktop"
    echo "3. Start using the community notes tools!"
    echo ""
    echo "To start the server manually:"
    echo "  source venv/bin/activate"
    echo "  python server.py"
    echo ""
else
    echo ""
    echo "ERROR: Data ingestion failed!"
    echo "Please check the error messages above."
    exit 1
fi
