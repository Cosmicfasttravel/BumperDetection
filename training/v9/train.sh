#!/bin/bash
set -eux
trap 'echo "Error on line $LINENO"' ERR

python3.10 -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install ultralytics roboflow opencv-python
pip install python-dotenv

./venv/bin/python3 train.py
