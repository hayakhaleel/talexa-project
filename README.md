Talexa
Talexa is an AI-powered lecture generation pipeline that turns source material into presentation assets. It combines document processing, slide generation, subtitle creation, speech generation, cursor planning, and talking-head video output in one workflow.

What It Does
Extracts and summarizes content from PDFs
Converts summaries into LaTeX and slide decks
Generates subtitles and translated subtitle JSON
Produces narration audio
Builds cursor movement data for slides
Creates talking-head video output
Serves a simple web interface for uploads and generation
Project Structure
Agents/ - core AI agents for each stage
PIPELINE/ - orchestration, preprocessing, session handling, and assembly
Prompts/ - prompt templates used by the agents
app/ - web interface, static files, and database logic
Run Locally
Create a virtual environment
Install the requirements
Make sure Ollama is running and required models are available
Set your API keys for ElevenLabs and HeyGen
Start the project
Example:

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
bash run_talexa.sh
Requirements
Python 3
Ollama
FFmpeg
LaTeX toolchain
ELEVENLABS_API_KEY
HEYGEN_API_KEY
Notes
This repository is an active project and the pipeline is still evolving. Some modules are more polished than others, but the repo contains the full main workflow and supporting web app.
