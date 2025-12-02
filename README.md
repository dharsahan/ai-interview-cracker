# Parakeet-like AI Interview Copilot

This is a proof-of-concept AI Interview Assistant similar to Parakeet AI. It features:
- **Audio Transcription**: Listens to your microphone (and system audio if configured) to transcribe interview questions.
- **AI Answers**: Uses an LLM (OpenAI GPT) to generate real-time answers.
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

### Python Dependencies
Install the required python packages:
```bash
pip install -r requirements.txt
```

## Configuration
Create a `.env` file in the root directory or enter your API key in the UI.
```
OPENAI_API_KEY=sk-...
```

## Running the App
Run the Streamlit application:
```bash
streamlit run src/app.py
```

## Troubleshooting
- **Audio Errors**: If you encounter PyAudio errors, ensure `portaudio` is installed correctly. In the sandbox or headless environments, use the "Mock" mode.
- **OCR Errors**: Ensure `tesseract` is installed and in your PATH.

## Disclaimer
This tool is for educational purposes and interview preparation (mock interviews). Using it during a real interview may violate company policies or ethical guidelines.
