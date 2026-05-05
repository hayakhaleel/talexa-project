#!/bin/bash

set -e

echo "🚀 Starting Talexa setup + run..."

echo "📦 Installing system dependencies..."
apt update
apt install -y ffmpeg zip unzip curl

echo "🐍 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo "🧠 Installing Ollama..."
if ! command -v ollama &> /dev/null
then
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo "⚙️ Starting Ollama..."
pkill ollama || true
ollama serve &
sleep 10

echo "📥 Pulling models..."

ollama pull qwen2.5:7b
ollama pull qwen2.5vl:7b
ollama pull qwen3-vl:8b

if [ -z "$ELEVENLABS_API_KEY" ]; then
  echo "❌ ELEVENLABS_API_KEY is not set"
  exit 1
fi

if [ -z "$HEYGEN_API_KEY" ]; then
  echo "❌ HEYGEN_API_KEY is not set"
  exit 1
fi

echo "🔑 API keys detected"

echo "🎬 Running Talexa pipeline..."

python -u PIPELINE/run_pipeline.py

echo "✅ Talexa finished!"
