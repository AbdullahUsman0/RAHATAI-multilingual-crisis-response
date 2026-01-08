"""
Whisper-based Speech-to-Text for Roman Urdu and Multilingual Support
Whisper wrapper for audio transcription
Supports: English, Urdu, Roman-Urdu
"""
import sys
from pathlib import Path
import os

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add FFmpeg to PATH if it exists (for Windows)
ffmpeg_path = r"C:\ffmpeg\ffmpeg-8.0.1-essentials_build\bin"
if os.path.exists(ffmpeg_path) and ffmpeg_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + ffmpeg_path

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from typing import Optional, Union, List
import io

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️  Whisper not installed. Install with: pip install openai-whisper")

class WhisperTranscriber:
    """
    Whisper-based Speech-to-Text Transcriber
    Supports multilingual transcription including Roman Urdu
    """
    
    def __init__(self, model_size: str = "base", device: str = None):
        """
        Initialize Whisper transcriber
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
                       "base" is good balance of speed and accuracy
            device: Device to use ("cpu", "cuda", or None for auto)
        """
        if not WHISPER_AVAILABLE:
            raise ImportError(
                "Whisper not installed. Install with: pip install openai-whisper\n"
                "For GPU support: pip install openai-whisper[gpu]"
            )
        
        self.model_size = model_size
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model"""
        print(f"Loading Whisper model: {self.model_size}...")
        try:
            self.model = whisper.load_model(self.model_size, device=self.device)
            print(f"✓ Whisper model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}")
    
    def transcribe(
        self, 
        audio: Union[str, Path, np.ndarray, bytes],
        language: Optional[str] = None,
        task: str = "transcribe",
        initial_prompt: Optional[str] = None
    ) -> dict:
        """
        Transcribe audio to text
        
        Args:
            audio: Audio file path, numpy array, or bytes
            language: Language code (e.g., "en", "ur", "hi") or None for auto-detect
                      For Roman Urdu, use "ur" or None (Whisper handles it well)
            task: "transcribe" or "translate" (translate to English)
            initial_prompt: Optional prompt to guide transcription (useful for Roman Urdu)
            
        Returns:
            dict: Transcription result with keys:
                - text: Transcribed text
                - language: Detected language
                - segments: List of segments with timestamps
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Initialize temp_file_created to track if we need cleanup
        temp_file_created = False
        audio_path = None
        
        # Handle different audio input types
        if isinstance(audio, (str, Path)):
            audio_path = str(audio)
        elif isinstance(audio, bytes):
            # Save bytes to temporary file
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', mode='wb') as tmp:
                tmp.write(audio)
                tmp.flush()  # Ensure data is written to disk
                audio_path = tmp.name
                temp_file_created = True
            # File is closed when exiting 'with' block, but we keep it for Whisper to read
            
            # Ensure file exists and is readable
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Temporary audio file not created: {audio_path}")
        elif isinstance(audio, np.ndarray):
            # Save numpy array to temporary file
            import tempfile
            import soundfile as sf
            import os
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', mode='wb') as tmp:
                sf.write(tmp.name, audio, 16000)  # Assume 16kHz sample rate
                audio_path = tmp.name
                temp_file_created = True
            # Ensure file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Temporary audio file not created: {audio_path}")
        
        # For Roman Urdu, use Urdu language code or auto-detect
        # Whisper handles Roman Urdu well when language is set to "ur" or auto-detected
        try:
            if language is None:
                # Auto-detect language (works well for Roman Urdu)
                result = self.model.transcribe(
                    audio_path,
                    task=task,
                    initial_prompt=initial_prompt,
                    language=None  # Auto-detect
                )
            else:
                result = self.model.transcribe(
                    audio_path,
                    language=language,
                    task=task,
                    initial_prompt=initial_prompt
                )
        except FileNotFoundError as e:
            # Check if it's an ffmpeg issue
            error_msg = str(e).lower()
            if 'ffmpeg' in error_msg or 'winerror 2' in error_msg or 'cannot find the file' in error_msg:
                raise FileNotFoundError(
                    "FFmpeg is not installed or not in PATH. Whisper requires FFmpeg to process audio.\n"
                    "Install FFmpeg:\n"
                    "1. Using Chocolatey: choco install ffmpeg\n"
                    "2. Or download from: https://www.gyan.dev/ffmpeg/builds/\n"
                    "3. Add ffmpeg/bin to your system PATH\n"
                    "4. Restart your terminal/IDE after installation\n"
                    "See INSTALL_FFMPEG.md for detailed instructions."
                ) from e
            raise
        finally:
            # Clean up temporary file if we created one
            if temp_file_created and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except:
                    pass  # Ignore cleanup errors
        
        return result
    
    def transcribe_audio_file(self, audio_path: Union[str, Path]) -> str:
        """
        Simple transcription - returns just the text
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            str: Transcribed text
        """
        result = self.transcribe(audio_path)
        return result["text"].strip()
    
    def transcribe_for_rag(
        self, 
        audio: Union[str, Path, np.ndarray, bytes],
        language: Optional[str] = None
    ) -> str:
        """
        Transcribe audio specifically for RAG queries
        Optimized for question-answering scenarios
        
        Args:
            audio: Audio input
            language: Language code or None for auto-detect
            
        Returns:
            str: Transcribed text ready for RAG query
        """
        # Use prompt to guide transcription for questions
        prompt = "This is a question about disaster response, emergency procedures, or crisis management."
        
        result = self.transcribe(
            audio,
            language=language,
            task="transcribe",
            initial_prompt=prompt
        )
        
        text = result["text"].strip()
        
        # Clean up common transcription errors for Roman Urdu
        text = self._clean_roman_urdu_text(text)
        
        return text
    
    def _clean_roman_urdu_text(self, text: str) -> str:
        """
        Clean and normalize Roman Urdu text from transcription
        
        Args:
            text: Raw transcribed text
            
        Returns:
            str: Cleaned text
        """
        # Common Roman Urdu patterns that might need cleaning
        # This is a basic implementation - can be extended
        
        # Remove extra spaces
        text = " ".join(text.split())
        
        # Fix common transcription errors
        replacements = {
            " kya ": " kya ",  
            " hai ": " hai ",
            " mein ": " mein ",
            " se ": " se ",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text.strip()

def transcribe_audio_file(audio_path: Union[str, Path], model_size: str = "base") -> str:
    """
    Convenience function to transcribe an audio file
    
    Args:
        audio_path: Path to audio file
        model_size: Whisper model size
        
    Returns:
        str: Transcribed text
    """
    transcriber = WhisperTranscriber(model_size=model_size)
    return transcriber.transcribe_audio_file(audio_path)

