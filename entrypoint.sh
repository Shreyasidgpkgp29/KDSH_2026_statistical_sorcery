#!/bin/bash

ollama serve &

echo "Waiting for Ollama..."
while ! curl -s http://localhost:11434/api/tags > /dev/null; do
    sleep 2
done

echo "Pulling model..."
ollama pull mistral 

echo "Pulling mxbai-embed-large model..."
ollama pull mxbai-embed-large

echo "Starting KDSH Prediction..."

python3 src/main.py