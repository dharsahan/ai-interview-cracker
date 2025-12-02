# Parakeet-like AI Interview Copilot

This is a proof-of-concept AI Interview Assistant similar to Parakeet AI. It features:
- **Audio Transcription**: Listens to your microphone (and system audio if configured) to transcribe interview questions.
- **AI Answers**: Uses a local LLM (Ollama) to generate real-time answers.
- **Visual Support**: Can capture the screen to help with coding questions (OCR).

## Prerequisites

### System Dependencies
You need to install system-level dependencies for audio and OCR.

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install python3-pyaudio portaudio19-dev tesseract-ocr
```

**MacOS:**
```bash
brew install portaudio tesseract
```

### Ollama Installation
Install Ollama to run local language models:

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**MacOS:**
```bash
brew install ollama
```

**Windows:**
Download from [ollama.com](https://ollama.com/download)

After installation, pull a model (e.g., llama3.2):
```bash
ollama pull llama3.2
```

### Python Dependencies
Install the required python packages:
```bash
pip install -r requirements.txt
```

## Configuration
Create a `.env` file in the root directory to customize Ollama settings (optional):
```
OLLAMA_MODEL=llama3.2
OLLAMA_HOST=http://localhost:11434
```

## Running the App

1. Start the Ollama server (if not already running):
```bash
ollama serve
```

2. Run the Streamlit application:
```bash
streamlit run src/app.py
```

## Troubleshooting
- **Ollama Connection Errors**: Ensure Ollama is running with `ollama serve` and accessible at the configured host (default: http://localhost:11434).
- **Model Not Found**: Pull the required model with `ollama pull <model_name>` (e.g., `ollama pull llama3.2`).
- **Audio Errors**: If you encounter PyAudio errors, ensure `portaudio` is installed correctly. In the sandbox or headless environments, use the "Mock" mode.
- **OCR Errors**: Ensure `tesseract` is installed and in your PATH.

## Disclaimer
This tool is for educational purposes and interview preparation (mock interviews). Using it during a real interview may violate company policies or ethical guidelines.
