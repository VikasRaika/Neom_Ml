# app/services/transcription.py
import os
import threading
import tempfile
from typing import Optional, Dict, Any

# Lazy-load Whisper (thread-safe)
_model = None
_model_lock = threading.Lock()

def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                import whisper  # import at first use
                _model = whisper.load_model("base")  # tiny/base/small/medium/large
    return _model

def transcribe_wav(path: str) -> str:
    """
    Transcribe an audio file using OpenAI Whisper.
    Supports .wav/.mp3/.m4a, etc (via ffmpeg).
    """
    model = _get_model()
    result = model.transcribe(path)
    text = (result.get("text") or "").strip()
    return text if text else " "

def transcribe_audio(file_bytes: Optional[bytes] = None) -> Dict[str, Any]:
    """
    Accept raw bytes from an UploadFile and return {"transcript": "..."}.
    Writes to a temporary file that ffmpeg/whisper can read.
    """
    if not file_bytes:
        return {"transcript": ""}

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            tmp_path = tmp.name

        text = transcribe_wav(tmp_path)
        return {"transcript": text}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

# Backward-compatible alias if anything imports a generic name
def transcribe(file_bytes: Optional[bytes] = None) -> Dict[str, Any]:
    return transcribe_audio(file_bytes)
