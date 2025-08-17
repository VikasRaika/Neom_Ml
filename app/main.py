# app/main.py
from fastapi import FastAPI, UploadFile, File
from typing import List
import numpy as np

from app.models import (
    TranscribeResponse,
    SummariseRequest,
    SummariseResponse,
    MatchRequest,
    MatchResponse,
)
from app.services.transcription import transcribe_wav
from app.services.topics import extract_topics_and_summary
from app.services.vectorize import make_topic_vector, fuse_vectors
from app.services.match import load_users, get_default_user_pair, compute_match
from app.utils.io import load_users


app = FastAPI(title="Neom ML")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/transcribe")
def transcribe(file: UploadFile = File(...)):
    # Save to disk then run whisper; (your current implementation)
    import tempfile, shutil, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        text = transcribe_wav(tmp_path)
        return {"transcript": text}
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# --------- /summarise ---------

@app.post("/summarise", response_model=SummariseResponse)
def summarise(req: SummariseRequest):
    topics, weights, summary = extract_topics_and_summary(req.transcript, top_k=req.top_k)

    # Build a single 384-d topic embedding from the phrases
    topic_vec = make_topic_vector(topics)
    topic_embedding = topic_vec.tolist() if topic_vec is not None else None

    return SummariseResponse(
        topics=topics,
        summary=summary,
        topic_embedding=topic_embedding,
    )


# --------- /match ---------

@app.post("/match", response_model=MatchResponse)
def match(req: MatchRequest):
    # Load two users from the sample JSON
    users = load_users()
    uid1, uid2 = get_default_user_pair(users)
    u1_traits = np.array(users[uid1]["traits"], dtype=float)
    u2_traits = np.array(users[uid2]["traits"], dtype=float)

    # Topic embedding (same content vector, different per-user scales)
    topic_vec = make_topic_vector(req.topics)  # (384,)

    result = compute_match(
        user1_traits=u1_traits,
        user2_traits=u2_traits,
        topic_vec=topic_vec,
        scale1=req.topic_scale_user1,
        scale2=req.topic_scale_user2,
    )
    # add which users were compared
    result["detail"] = f"{result['detail']}; compared users={uid1} vs {uid2}"
    return MatchResponse(**result)