import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from audio import AudioTranscriber
from llm import LLMClient
from vision import ScreenCapturer

class TestAudioTranscriber(unittest.TestCase):
    def test_mock_mode_initialization(self):
        # Force mock mode
        transcriber = AudioTranscriber(mock_mode=True)
        self.assertTrue(transcriber.mock_mode)

    def test_mock_listening(self):
        transcriber = AudioTranscriber(mock_mode=True)
        transcriber.start_listening()
        self.assertTrue(transcriber.is_recording)
        transcriber.stop_listening()
        self.assertFalse(transcriber.is_recording)

class TestLLMClient(unittest.TestCase):
    def test_mock_llm(self):
        client = LLMClient(api_key=None) # No key -> mock
        answer = client.get_answer("What is Python?")
        self.assertIn("[MOCK AI ANSWER]", answer)

class TestScreenCapturer(unittest.TestCase):
    @patch('src.vision.mss.mss')
    def test_screen_capture_headless_check(self, mock_mss):
        # In this env, DISPLAY might be missing, so it should return error or handle it.
        # But we want to test the class instantiation at least.
        capturer = ScreenCapturer()
        self.assertIsNotNone(capturer)

if __name__ == '__main__':
    unittest.main()
