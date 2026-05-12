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
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
bash run_talexa.sh
```

## Requirements

- Python 3
- Ollama
- FFmpeg
- LaTeX toolchain
- `ELEVENLABS_API_KEY`
- `HEYGEN_API_KEY`

## Status

This project is actively being developed. The repository contains the main workflow and supporting web app, with some components more mature than others.
