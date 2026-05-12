# Talexa

Talexa is an AI-powered lecture generation pipeline that transforms source material into presentation-ready assets. It combines document understanding, slide generation, subtitle creation, speech synthesis, cursor planning, and talking-head video generation in a single workflow.

## Overview

Talexa is designed to automate the process of turning educational content into polished lecture media. The project includes both a backend pipeline and a lightweight web interface for managing uploads and generation.

## Features

- PDF content extraction and summarization
- LaTeX and slide deck generation
- Subtitle and translation pipeline
- Narration audio generation
- Cursor movement planning for slides
- Talking-head video generation
- Simple web-based upload and generation flow

## Project Structure

```text
Agents/      Core AI agents for each processing stage
PIPELINE/    Workflow orchestration, preprocessing, and assembly
Prompts/     Prompt templates used by the agents
app/         Web interface, static assets, and database logic
```

## Quick Start

```bash
apt update
apt install -y git curl wget unzip zip zstd ffmpeg sox libgl1 libglib2.0-0 libsndfile1 build-essential python3-dev texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-xetex nano

cd /workspace/talexa-project
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

python scripts/download_nllb_translation_model.py

ollama pull qwen2.5:7b
ollama pull qwen2.5vl:7b
ollama pull qwen3
ollama pull qwen3-vl:8b

PORT=8000 python app/app.py
```

## Requirements

- Python 3
- Virtual environment (`venv`)
- Ollama
- FFmpeg
- LaTeX toolchain
- System packages: `git`, `curl`, `wget`, `unzip`, `zip`, `zstd`, `sox`, `libgl1`, `libglib2.0-0`, `libsndfile1`, `build-essential`, `python3-dev`, `texlive-latex-base`, `texlive-latex-extra`, `texlive-fonts-recommended`, `texlive-xetex`, `nano`
- `ELEVENLABS_API_KEY`
- `HEYGEN_API_KEY`

## Python Dependencies

The Python packages are listed in [`requirements.txt`](/Users/hayakhaleel/Downloads/talexa-project-main%202/requirements.txt). It includes the packages used by the current codebase plus the runtime packages needed for the translation model download script and your local app workflow.

## Ollama Models

Talexa expects these Ollama models to be available locally:

```bash
ollama pull qwen2.5:7b
ollama pull qwen2.5vl:7b
ollama pull qwen3
ollama pull qwen3-vl:8b
```

## Status

This project is actively being developed. The repository contains the main workflow and supporting web app, with some components more mature than others.
