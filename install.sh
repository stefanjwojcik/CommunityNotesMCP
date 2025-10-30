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

# Check if data files exist
echo "Checking for data files..."
if [ ! -f "data/notes-00000.tsv" ]; then
    echo "ERROR: data/notes-00000.tsv not found!"
    echo "Please ensure the data files are in the data/ directory."
    exit 1
fi

if [ ! -f "data/noteStatusHistory-00000.tsv" ]; then
    echo "ERROR: data/noteStatusHistory-00000.tsv not found!"
    echo "Please ensure the data files are in the data/ directory."
    exit 1
fi
echo "Data files found!"
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
