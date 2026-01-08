"""
Speech-to-Text module for RahatAI
Supports Whisper-based transcription for multilingual audio including Roman Urdu
"""

from Scripts.speech.whisper_transcriber import WhisperTranscriber, transcribe_audio_file

__all__ = ['WhisperTranscriber', 'transcribe_audio_file']



