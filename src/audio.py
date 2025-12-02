import os
import time
import speech_recognition as sr
from threading import Thread, Event
from queue import Queue

class AudioTranscriber:
    def __init__(self, mock_mode=False):
        self.mock_mode = mock_mode
        self.recognizer = sr.Recognizer()
        self.audio_queue = Queue()
        self.stop_event = Event()
        self.is_recording = False

        # Check if we are in a headless environment (simple heuristic)
        if not self.mock_mode:
            try:
                import pyaudio
                self.pyaudio_available = True
            except ImportError:
                print("PyAudio not available. Falling back to Mock Mode.")
                self.mock_mode = True
                self.pyaudio_available = False

    def start_listening(self):
        """Starts a background thread to listen for audio."""
        if self.is_recording:
            return

        self.is_recording = True
        self.stop_event.clear()

        if self.mock_mode:
            self.thread = Thread(target=self._mock_listen_loop)
        else:
            try:
                # We need to check for microphones
                mics = sr.Microphone.list_microphone_names()
                if not mics:
                    print("No microphone found. Falling back to Mock Mode.")
                    self.mock_mode = True
                    self.thread = Thread(target=self._mock_listen_loop)
                else:
                    self.thread = Thread(target=self._listen_loop)
            except Exception as e:
                print(f"Error accessing microphone: {e}. Falling back to Mock Mode.")
                self.mock_mode = True
                self.thread = Thread(target=self._mock_listen_loop)

        self.thread.start()

    def stop_listening(self):
        """Stops the listening thread."""
        self.is_recording = False
        self.stop_event.set()
        if hasattr(self, 'thread'):
            self.thread.join()

    def _listen_loop(self):
        """Real listening loop using SpeechRecognition."""
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            while not self.stop_event.is_set():
                try:
                    # Listen for a short phrase (non-blocking logic effectively handled by short phrase_time_limit or just iterating)
                    # We use a short timeout to check stop_event frequently
                    try:
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=10)
                        self.audio_queue.put(audio)
                    except sr.WaitTimeoutError:
                        continue
                except Exception as e:
                    print(f"Error in listen loop: {e}")
                    break

    def _mock_listen_loop(self):
        """Simulates listening by generating dummy text periodically."""
        mock_phrases = [
            "Can you explain the difference between a process and a thread?",
            "What is a REST API?",
            "How do you handle dependency injection in Python?",
            "Describe the CAP theorem.",
            "Write a function to reverse a linked list."
        ]
        import random
        while not self.stop_event.is_set():
            time.sleep(5)  # Simulate pause between questions
            phrase = random.choice(mock_phrases)
            self.audio_queue.put(("text", phrase)) # Special tuple for mock text

    def get_transcript(self):
        """
        Process the queue and return new text.
        Returns a list of strings (newly transcribed segments).
        """
        new_transcripts = []
        while not self.audio_queue.empty():
            item = self.audio_queue.get()

            if isinstance(item, tuple) and item[0] == "text":
                # Mock text
                new_transcripts.append(item[1])
            else:
                # Real audio
                try:
                    # Using Google Speech Recognition as it doesn't require API key (limited use)
                    # or could use Whisper if installed.
                    text = self.recognizer.recognize_google(item)
                    new_transcripts.append(text)
                except sr.UnknownValueError:
                    pass # Could not understand audio
                except sr.RequestError as e:
                    new_transcripts.append(f"[Error: {e}]")

        return new_transcripts
