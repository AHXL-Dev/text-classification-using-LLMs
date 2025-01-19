#!/bin/bash

# Exit immediately if any command fails
set -e

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

echo "Starting Ollama server in the background..."
ollama serve &

echo "Pulling the Llama 3 (8B) model..."
ollama pull llama3:8b

echo "Setup complete! Ollama is running, and the model is ready."

