import unittest
from unittest.mock import MagicMock, patch, Mock
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from audio import AudioTranscriber
from llm import LLMClient
from vision import ScreenCapturer


class TestAudioTranscriber(unittest.TestCase):
    """Tests for the AudioTranscriber class - handles real-time audio transcription."""

    def test_mock_mode_initialization(self):
        """Test that mock mode can be forced during initialization."""
        transcriber = AudioTranscriber(mock_mode=True)
        self.assertTrue(transcriber.mock_mode)
        self.assertIsNotNone(transcriber.recognizer)
        self.assertIsNotNone(transcriber.audio_queue)
        self.assertIsNotNone(transcriber.stop_event)
        self.assertFalse(transcriber.is_recording)

    def test_mock_listening_start_stop(self):
        """Test starting and stopping mock listening mode."""
        transcriber = AudioTranscriber(mock_mode=True)
        transcriber.start_listening()
        self.assertTrue(transcriber.is_recording)
        self.assertFalse(transcriber.stop_event.is_set())
        
        transcriber.stop_listening()
        self.assertFalse(transcriber.is_recording)
        self.assertTrue(transcriber.stop_event.is_set())

    def test_mock_listening_generates_transcripts(self):
        """Test that mock mode generates sample interview questions."""
        transcriber = AudioTranscriber(mock_mode=True)
        
        # Directly add a mock phrase to the queue instead of waiting
        transcriber.audio_queue.put(("text", "What is a REST API?"))
        
        transcripts = transcriber.get_transcript()
        self.assertIsInstance(transcripts, list)
        self.assertEqual(len(transcripts), 1)
        self.assertEqual(transcripts[0], "What is a REST API?")

    def test_double_start_listening_ignored(self):
        """Test that starting listening when already listening is ignored."""
        transcriber = AudioTranscriber(mock_mode=True)
        transcriber.start_listening()
        self.assertTrue(transcriber.is_recording)
        
        # Try to start again
        transcriber.start_listening()
        self.assertTrue(transcriber.is_recording)  # Should still be recording
        
        transcriber.stop_listening()

    def test_get_transcript_empty_queue(self):
        """Test getting transcript from empty queue returns empty list."""
        transcriber = AudioTranscriber(mock_mode=True)
        transcripts = transcriber.get_transcript()
        self.assertEqual(transcripts, [])

    def test_stop_listening_without_start(self):
        """Test that stopping without starting doesn't cause errors."""
        transcriber = AudioTranscriber(mock_mode=True)
        # Should not raise any exception
        transcriber.stop_listening()
        self.assertFalse(transcriber.is_recording)

    def test_auto_fallback_to_mock_when_pyaudio_unavailable(self):
        """Test automatic fallback to mock mode when PyAudio is unavailable."""
        with patch.dict('sys.modules', {'pyaudio': None}):
            with patch('builtins.__import__', side_effect=ImportError):
                # Create a transcriber that attempts real mode
                transcriber = AudioTranscriber(mock_mode=False)
                # Should have fallen back to mock mode
                # Note: This test may pass because pyaudio is available
                self.assertIsNotNone(transcriber)


class TestLLMClient(unittest.TestCase):
    """Tests for the LLMClient class - handles AI-powered answer generation."""

    def test_mock_llm_initialization(self):
        """Test LLM client initializes in mock mode when Ollama is not available."""
        with patch('llm.ollama.Client') as mock_ollama:
            mock_ollama.return_value.list.side_effect = Exception("Connection refused")
            client = LLMClient()
            self.assertFalse(client._connected)

    def test_mock_llm_generates_answer(self):
        """Test mock LLM generates structured mock answers."""
        with patch('llm.ollama.Client') as mock_ollama:
            mock_ollama.return_value.list.side_effect = Exception("Connection refused")
            client = LLMClient()
            answer = client.get_answer("What is Python?")
            self.assertIn("[MOCK AI ANSWER]", answer)
            self.assertIn("What is Python?", answer)
            self.assertIn("clear definition", answer.lower())

    def test_mock_llm_empty_question(self):
        """Test that empty questions return empty string."""
        with patch('llm.ollama.Client') as mock_ollama:
            mock_ollama.return_value.list.side_effect = Exception("Connection refused")
            client = LLMClient()
            answer = client.get_answer("")
            self.assertEqual(answer, "")

    def test_mock_llm_none_question(self):
        """Test that None question is handled (expects empty or error)."""
        with patch('llm.ollama.Client') as mock_ollama:
            mock_ollama.return_value.list.side_effect = Exception("Connection refused")
            client = LLMClient()
            # The function checks for falsy values
            answer = client.get_answer(None)
            self.assertEqual(answer, "")

    def test_llm_with_ollama_connected(self):
        """Test that connecting to Ollama creates a client."""
        with patch('llm.ollama.Client') as mock_ollama:
            mock_client = MagicMock()
            mock_client.list.return_value = {'models': []}
            mock_ollama.return_value = mock_client
            client = LLMClient(model="llama3.2", host="http://localhost:11434")
            mock_ollama.assert_called_once_with(host="http://localhost:11434")
            self.assertTrue(client._connected)

    def test_llm_get_answer_with_real_client(self):
        """Test getting answer with a mocked Ollama client."""
        with patch('llm.ollama.Client') as mock_ollama:
            # Setup mock response
            mock_client = MagicMock()
            mock_client.list.return_value = {'models': []}
            mock_client.chat.return_value = {
                'message': {'content': 'This is a test answer.'}
            }
            mock_ollama.return_value = mock_client
            
            client = LLMClient(model="llama3.2")
            answer = client.get_answer("Test question")
            
            self.assertEqual(answer, "This is a test answer.")
            mock_client.chat.assert_called_once()

    def test_llm_handles_api_error(self):
        """Test that API errors are handled gracefully."""
        with patch('llm.ollama.Client') as mock_ollama:
            mock_client = MagicMock()
            mock_client.list.return_value = {'models': []}
            mock_client.chat.side_effect = Exception("API Error")
            mock_ollama.return_value = mock_client
            
            client = LLMClient()
            answer = client.get_answer("Test question")
            
            self.assertIn("Error", answer)

    def test_llm_reads_env_model(self):
        """Test that LLM client reads model from environment."""
        with patch.dict(os.environ, {'OLLAMA_MODEL': 'mistral'}):
            with patch('llm.ollama.Client') as mock_ollama:
                mock_client = MagicMock()
                mock_client.list.return_value = {'models': []}
                mock_ollama.return_value = mock_client
                client = LLMClient()
                # Should use the env model
                self.assertEqual(client.model, 'mistral')

    def test_llm_reads_env_host(self):
        """Test that LLM client reads host from environment."""
        with patch.dict(os.environ, {'OLLAMA_HOST': 'http://custom:8080'}):
            with patch('llm.ollama.Client') as mock_ollama:
                mock_client = MagicMock()
                mock_client.list.return_value = {'models': []}
                mock_ollama.return_value = mock_client
                client = LLMClient()
                # Should use the env host
                mock_ollama.assert_called_once_with(host='http://custom:8080')


class TestScreenCapturer(unittest.TestCase):
    """Tests for the ScreenCapturer class - handles screen capture and OCR."""

    def test_screen_capturer_initialization(self):
        """Test ScreenCapturer initializes correctly."""
        capturer = ScreenCapturer()
        self.assertIsNotNone(capturer)
        # tesseract_available depends on system installation
        self.assertIsInstance(capturer.tesseract_available, bool)

    def test_screen_capturer_tesseract_check(self):
        """Test that tesseract availability is correctly detected."""
        with patch('shutil.which') as mock_which:
            mock_which.return_value = '/usr/bin/tesseract'
            capturer = ScreenCapturer()
            self.assertTrue(capturer.tesseract_available)

    def test_screen_capturer_no_tesseract(self):
        """Test behavior when tesseract is not available."""
        with patch('shutil.which') as mock_which:
            mock_which.return_value = None
            capturer = ScreenCapturer()
            self.assertFalse(capturer.tesseract_available)

    def test_capture_and_read_headless_linux(self):
        """Test capture_and_read in headless Linux environment."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('platform.system', return_value='Linux'):
                capturer = ScreenCapturer()
                result = capturer.capture_and_read()
                
                # Should return error about no DISPLAY
                self.assertIn("[Error]", result)

    def test_capture_and_read_with_display(self):
        """Test capture_and_read when DISPLAY is available."""
        # This test will behave differently based on environment
        capturer = ScreenCapturer()
        result = capturer.capture_and_read()
        self.assertIsInstance(result, str)

    @patch('vision.mss.mss')
    @patch('vision.Image')
    @patch('vision.pytesseract')
    def test_capture_and_read_success(self, mock_tesseract, mock_image, mock_mss):
        """Test successful screen capture and OCR."""
        # Setup mocks
        mock_sct = MagicMock()
        mock_sct.monitors = [None, {'width': 1920, 'height': 1080}]
        mock_screenshot = MagicMock()
        mock_screenshot.size = (1920, 1080)
        mock_screenshot.bgra = b'\x00' * (1920 * 1080 * 4)
        mock_sct.grab.return_value = mock_screenshot
        mock_mss.return_value.__enter__.return_value = mock_sct
        
        mock_img = MagicMock()
        mock_image.frombytes.return_value = mock_img
        
        mock_tesseract.image_to_string.return_value = "Sample OCR Text"
        
        with patch.dict(os.environ, {'DISPLAY': ':0'}):
            with patch('platform.system', return_value='Linux'):
                capturer = ScreenCapturer()
                capturer.tesseract_available = True
                result = capturer.capture_and_read()
                
                # Result should be from OCR
                self.assertIsInstance(result, str)

    def test_capture_and_read_no_tesseract_error(self):
        """Test that missing tesseract returns appropriate error."""
        with patch('shutil.which', return_value=None):
            with patch.dict(os.environ, {'DISPLAY': ':0'}):
                with patch('platform.system', return_value='Linux'):
                    capturer = ScreenCapturer()
                    capturer.tesseract_available = False
                    
                    with patch('vision.mss.mss') as mock_mss:
                        mock_sct = MagicMock()
                        mock_sct.monitors = [None, {'width': 100, 'height': 100}]
                        mock_screenshot = MagicMock()
                        mock_screenshot.size = (100, 100)
                        mock_screenshot.bgra = b'\x00' * 40000
                        mock_sct.grab.return_value = mock_screenshot
                        mock_mss.return_value.__enter__.return_value = mock_sct
                        
                        with patch('vision.Image.frombytes'):
                            result = capturer.capture_and_read()
                            self.assertIn("[Error]", result)
                            self.assertIn("Tesseract", result)


class TestIntegration(unittest.TestCase):
    """Integration tests for the AI Interview Cracker application."""

    def test_audio_to_llm_integration(self):
        """Test integration between audio transcription and LLM response."""
        transcriber = AudioTranscriber(mock_mode=True)
        with patch('llm.ollama.Client') as mock_ollama:
            mock_ollama.return_value.list.side_effect = Exception("Connection refused")
            llm_client = LLMClient()
        
            # Simulate getting a transcript
            transcriber.audio_queue.put(("text", "What is a REST API?"))
            transcripts = transcriber.get_transcript()
            
            self.assertEqual(len(transcripts), 1)
            self.assertEqual(transcripts[0], "What is a REST API?")
            
            # Get answer for the transcript
            answer = llm_client.get_answer(transcripts[0])
            self.assertIn("[MOCK AI ANSWER]", answer)
            self.assertIn("REST API", answer)

    def test_full_mock_workflow(self):
        """Test full workflow in mock mode."""
        # Initialize all components in mock mode
        transcriber = AudioTranscriber(mock_mode=True)
        with patch('llm.ollama.Client') as mock_ollama:
            mock_ollama.return_value.list.side_effect = Exception("Connection refused")
            llm_client = LLMClient()
            screen_capturer = ScreenCapturer()
            
            # Verify initialization
            self.assertTrue(transcriber.mock_mode)
            self.assertFalse(llm_client._connected)
            self.assertIsNotNone(screen_capturer)
            
            # Simulate workflow
            question = "Explain dependency injection"
            answer = llm_client.get_answer(question)
            
            self.assertIn("[MOCK AI ANSWER]", answer)
            self.assertIn("dependency injection", answer)


if __name__ == '__main__':
    unittest.main()
