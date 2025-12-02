import streamlit as st
import threading
import time
import os
from audio import AudioTranscriber
from llm import LLMClient
from vision import ScreenCapturer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Parakeet-like Interview Copilot", layout="wide")

# Initialize Session State
if 'transcriber' not in st.session_state:
    st.session_state.transcriber = AudioTranscriber(mock_mode=False) # Will auto-fallback
if 'llm' not in st.session_state:
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    st.session_state.llm = LLMClient(model=model, host=host)
if 'vision' not in st.session_state:
    st.session_state.vision = ScreenCapturer()
if 'transcript_history' not in st.session_state:
    st.session_state.transcript_history = []
if 'latest_answer' not in st.session_state:
    st.session_state.latest_answer = ""
if 'listening' not in st.session_state:
    st.session_state.listening = False

def toggle_listening():
    if st.session_state.listening:
        st.session_state.transcriber.stop_listening()
        st.session_state.listening = False
    else:
        st.session_state.transcriber.start_listening()
        st.session_state.listening = True

def capture_screen_action():
    text = st.session_state.vision.capture_and_read()
    if text:
        st.session_state.transcript_history.append(f"**[Screen Capture]**: {text}")
        answer = st.session_state.llm.get_answer(text)
        st.session_state.latest_answer = answer

# Sidebar
with st.sidebar:
    st.title("Settings")

    st.subheader("Ollama Configuration")
    model_input = st.text_input("Model Name", value=os.getenv("OLLAMA_MODEL", "llama3.2"))
    host_input = st.text_input("Ollama Host", value=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    
    if st.button("Update LLM Settings"):
        st.session_state.llm = LLMClient(model=model_input, host=host_input)
        if st.session_state.llm._connected:
            st.success("Connected to Ollama!")
        else:
            st.warning("Could not connect to Ollama. Running in Mock Mode.")

    st.divider()

    st.write("Status:")
    if st.session_state.transcriber.mock_mode:
        st.warning("Audio: Mock Mode (Microphone not detected/usable)")
    else:
        st.success("Audio: Real Mode")

    if st.session_state.llm._connected:
        st.success(f"LLM: Connected ({st.session_state.llm.model})")
    else:
        st.warning("LLM: Mock Mode (Ollama not running)")

# Main Layout
st.title("🦜 AI Interview Copilot")
st.markdown("Real-time transcription and AI assistance.")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Live Transcript")

    # Control Buttons
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Start/Stop Listening", type="primary" if not st.session_state.listening else "secondary"):
            toggle_listening()
            st.rerun()

    with btn_col2:
        if st.button("Capture Screen (Coding Qs)"):
            capture_screen_action()
            st.rerun()

    if st.session_state.listening:
        st.write("🔴 *Listening...*")

        # Poll for new transcripts
        new_text = st.session_state.transcriber.get_transcript()
        if new_text:
            for text in new_text:
                st.session_state.transcript_history.append(f"**[Audio]**: {text}")
                # Generate answer for the latest question automatically?
                # Or wait for user trigger? Let's do auto for the latest one.
                answer = st.session_state.llm.get_answer(text)
                st.session_state.latest_answer = answer
            st.rerun()

    # Display History
    chat_container = st.container(height=400)
    with chat_container:
        for line in st.session_state.transcript_history:
            st.markdown(line)

with col2:
    st.header("AI Suggested Answer")
    if st.session_state.latest_answer:
        st.info(st.session_state.latest_answer)
    else:
        st.write("Waiting for questions...")

# Auto-refresh loop for polling (Streamlit specific hack or using st.empty)
if st.session_state.listening:
    time.sleep(1)
    st.rerun()
