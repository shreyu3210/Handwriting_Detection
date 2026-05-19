#!/bin/bash

echo "Starting n8n containers..."
cd /home/shreyansh1812/n8n || exit

docker compose up -d

echo "Starting OCR backend..."
cd /home/shreyansh1812/Desktop/Projects/BM_Handwritten_ai || exit

source .venv/bin/activate

python app.py