import mss
import mss.tools
from PIL import Image
import pytesseract
import platform
import os

class ScreenCapturer:
    def __init__(self):
        # Check if tesseract is available
        self.tesseract_available = False
        try:
            # Simple check if tesseract is in path or callable
            # We assume it is if installed.
            # In a real app we might want to shutil.which("tesseract")
            import shutil
            if shutil.which("tesseract"):
                self.tesseract_available = True
        except Exception:
            pass

    def capture_and_read(self):
        """
        Captures the screen and performs OCR.
        Returns the text found on screen.
        """
        if platform.system() == "Linux" and not os.environ.get("DISPLAY"):
             return "[Error] Cannot capture screen: No DISPLAY environment variable found (Headless Mode)."

        try:
            with mss.mss() as sct:
                # Capture the primary monitor
                monitor = sct.monitors[1]
                screenshot = sct.grab(monitor)

                # Convert to PIL Image
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

                if self.tesseract_available:
                    text = pytesseract.image_to_string(img)
                    return text.strip()
                else:
                    return "[Error] Tesseract OCR not installed on system. Cannot read screen text."
        except Exception as e:
            return f"[Error] Screen capture failed: {e}"
