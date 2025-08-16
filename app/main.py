from fastapi import FastAPI, UploadFile, File, HTTPException
import tempfile, shutil, os

# services & models
from app.services.transcription import transcribe_wav
from app.services.topics import extract_topics_and_summary
from app.services.vectorize import make_topic_vector, fuse_vectors
from app.services.match import cosine_similarity, interpret

from app.models import (
    TranscribeResponse,
    SummariseRequest, SummariseResponse,
    MatchRequest, MatchResponse
)
from app.utils.io import load_users

app = FastAPI(title="Mini ML Pipeline", version="0.1.0")

# load users once at startup
USERS = load_users()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".wav", ".mp3", ".m4a")):
        raise HTTPException(status_code=400, detail="Upload an audio file (.wav/.mp3/.m4a).")
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        text = transcribe_wav(tmp_path)
        return TranscribeResponse(transcript=text)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

@app.post("/summarise", response_model=SummariseResponse)
async def summarise(req: SummariseRequest):
    topics, weights, summary = extract_topics_and_summary(req.transcript)
    return SummariseResponse(topics=topics, topic_weights=weights, summary=summary)

@app.post("/match", response_model=MatchResponse)
async def match(req: MatchRequest):
    # validate users
    if req.user_id_1 not in USERS or req.user_id_2 not in USERS:
        raise HTTPException(status_code=404, detail="One or both user IDs not found.")

    # topics -> vector
    if len(req.topics) != len(req.topic_weights):
        raise HTTPException(status_code=400, detail="topics and topic_weights length mismatch.")
    topic_vec = make_topic_vector(req.topics, req.topic_weights)

    # fuse psychometrics + topics
    u1 = fuse_vectors(USERS[req.user_id_1], topic_vec, alpha=req.alpha, beta=req.beta)
    u2 = fuse_vectors(USERS[req.user_id_2], topic_vec, alpha=req.alpha, beta=req.beta)

    # score + label
    score = cosine_similarity(u1, u2)
    label = interpret(score)
    detail = f"combined_dim={len(u1)}; alpha={req.alpha}, beta={req.beta}"

    return MatchResponse(score=round(score, 4), label=label, detail=detail)
