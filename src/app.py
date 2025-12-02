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
    api_key = os.getenv("OPENAI_API_KEY")
    st.session_state.llm = LLMClient(api_key=api_key)
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

    api_key_input = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    if api_key_input and api_key_input != st.session_state.llm.api_key:
        st.session_state.llm = LLMClient(api_key=api_key_input)
        st.success("API Key updated!")

    st.divider()

    st.write("Status:")
    if st.session_state.transcriber.mock_mode:
        st.warning("Audio: Mock Mode (Microphone not detected/usable)")
    else:
        st.success("Audio: Real Mode")

    if st.session_state.llm.client:
        st.success("LLM: Connected")
    else:
        st.warning("LLM: Mock Mode (No API Key)")

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
