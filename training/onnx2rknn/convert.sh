#!/bin/bash
clear

if ! command -v python3.10 &> /dev/null; then
    echo "Error: python3.10 is not installed on system."
    echo "Run: sudo dnf install python3.10"
    exit 1
fi

if [ -d "venv" ]; then
    if [[ $(./venv/bin/python --version 2>&1) != *"3.10"* ]]; then
        echo "Wiping old python sandbox..."
        rm -rf venv
    fi
fi

if [ ! -d "venv" ]; then
    echo "Creating local sandbox environment using Python 3.10"
    python3.10 -m venv venv
    source venv/bin/activate
    echo "Fetching core computational tools"
    pip install --upgrade pip wheel
    pip install "setuptools<81"
    
    pip install onnx==1.18.0
    
    pip install torch numpy==1.24.4 rknn-toolkit2
else
    source venv/bin/activate
fi

echo "Compiling asset graph"
./venv/bin/python3 convert.py

echo ""
echo "Press Enter to close."
read
