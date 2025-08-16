import os
import threading

# lazy-load the model once (thread-safe)
_model = None
_model_lock = threading.Lock()

def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                import whisper  # imported here to avoid slow import at startup
                # choose a size: tiny/base/small/medium/large
                _model = whisper.load_model("base")  # good balance of speed/quality
    return _model

def transcribe_wav(path: str) -> str:
    """
    Transcribe an audio file using OpenAI Whisper.
    Supports .wav/.mp3/.m4a, etc (via ffmpeg).
    """
    model = _get_model()
    try:
        result = model.transcribe(path)
        text = (result.get("text") or "").strip()
        return text if text else " "
    except Exception as e:
        # Bubble up a short error; FastAPI handler will return JSON
        raise RuntimeError(f"Transcription failed: {e}")
